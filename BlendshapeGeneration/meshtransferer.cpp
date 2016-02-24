#include "meshtransferer.h"
#include "triangle_gradient.h"

MeshTransferer::MeshTransferer()
{
  S0set = false;
  T0set = false;
}

MeshTransferer::~MeshTransferer()
{

}

void MeshTransferer::setSource(const BasicMesh &src) {
  S0 = src;
  computeS0grad();
  S0set = true;
}

void MeshTransferer::setTarget(const BasicMesh &tgt) {
  T0 = tgt;
  computeT0grad();
  T0set = true;
}

BasicMesh MeshTransferer::transfer(const BasicMesh &S1)
{
  if( !(S0set && T0set) ) {
    throw "S0 or T0 not set.";
  }
  // compute the deformation gradients of S1
  int nfaces = S1.NumFaces();
  vector<PhGUtils::Matrix3x3d> S1grad(nfaces);

  for(int j=0;j<nfaces;++j) {
    auto &V = S0grad[j];
    auto Vt = triangleGradient(S1, j);
    S1grad[j] = (Vt * V.inv()).transposed();
  }

  // delegate to transfer by deformation gradients
  return transfer(S1grad);
}

BasicMesh MeshTransferer::transfer(const vector<PhGUtils::Matrix3x3d> &S1grad)
{
  if( !(S0set && T0set) ) {
    throw "S0 or T0 not set.";
  }

  auto &S = S1grad;
  auto &T = T0grad;

  int nfaces = S0.NumFaces();
  int nverts = S0.NumVertices();

  // assemble sparse matrix A
  int nrowsA = nfaces * 3;
  int nsv = stationary_vertices.size();
  int nrowsC = nsv;
  int nrows = nrowsA + nrowsC;
  int ncols = nverts;
  int ntermsA = nfaces*9;
  int ntermsC = stationary_vertices.size();
  int nterms = ntermsA + ntermsC;

  using Tripletd = Eigen::Triplet<double>;
  using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;

  PhGUtils::Timer tmatrix;
  tmatrix.tic();

  vector<Tripletd> A_coeffs;
  A_coeffs.reserve(nterms);

  SparseMatrixd A(nrows, ncols);

  // fill in the deformation gradient part
  for(int i=0, ioffset=0;i<nfaces;++i) {
    /*
     * Ai:
     *     1 2 3 4 5 ... nfaces*3
     *     1 2 3 4 5 ... nfaces*3
     *     1 2 3 4 5 ... nfaces*3
     * Ai = reshape(Ai, 1, nfaces*9)
     *
     * Aj = reshape(repmat(S0.faces', 3, 1), 1, nfaces*9)
     * Av = reshape(cell2mat(T)', 1, nfaces*9)
     */
    auto f = S0.face(i);

    auto Ti = T[i];

    A_coeffs.push_back(Tripletd(ioffset, f[0], Ti(0)));
    A_coeffs.push_back(Tripletd(ioffset, f[1], Ti(1)));
    A_coeffs.push_back(Tripletd(ioffset, f[2], Ti(2)));
    ++ioffset;

    A_coeffs.push_back(Tripletd(ioffset, f[0], Ti(3)));
    A_coeffs.push_back(Tripletd(ioffset, f[1], Ti(4)));
    A_coeffs.push_back(Tripletd(ioffset, f[2], Ti(5)));
    ++ioffset;

    A_coeffs.push_back(Tripletd(ioffset, f[0], Ti(6)));
    A_coeffs.push_back(Tripletd(ioffset, f[1], Ti(7)));
    A_coeffs.push_back(Tripletd(ioffset, f[2], Ti(8)));
    ++ioffset;
  }

  const double w_stationary = 0.1;

  // fill in the lower part of A, stationary vertices part
  for(int i=0;i<nsv;++i) {
    A_coeffs.push_back(Tripletd(nrowsA+i, stationary_vertices[i], w_stationary));
  }

  A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());

  //ofstream fA("A.txt");
  //fA<<A;
  //fA.close();

  // fill in c matrix
  MatrixXd c(nrows, 3);
  for(int i=0;i<3;++i) {
    for(int j=0, joffset=0;j<nfaces;++j) {
      auto &Sj = S[j];
      c(joffset, i) = Sj(0, i); ++joffset;
      c(joffset, i) = Sj(1, i); ++joffset;
      c(joffset, i) = Sj(2, i); ++joffset;
    }
  }
  for(int i=0;i<3;++i) {
    for(int j=0, joffset=nrowsA;j<nsv;++j,++joffset) {
      auto vj = T0.vertex(stationary_vertices[j]);
      c(joffset, i) = vj[i] * w_stationary;
    }
  }

  vector<Tripletd> D_coeffs;
  D_coeffs.reserve(nrowsA);
  for(int i=0;i<nrowsA;++i) {
    int fidx = i / 3;
    D_coeffs.push_back(Tripletd(i, i, Ds[fidx]));
  }
  for(int i=nrowsA;i<nrows;++i) {
    D_coeffs.push_back(Tripletd(i, i, 1));
  }
  SparseMatrixd D(nrows, nrows);
  D.setFromTriplets(D_coeffs.begin(), D_coeffs.end());

  tmatrix.toc("constructing linear equations");

  PhGUtils::Timer tsolve;
  tsolve.tic();

  auto G = A;
  auto Gt = G.transpose();

  auto GtD = Gt * D;

  // compute GtDG
  auto GtDG = (GtD * G).pruned();

  // compute GtD * c
  auto GtDc = GtD * c;

  // solve for GtDG \ GtDc
  CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(GtDG);
  if(solver.info()!=Success) {
    cerr << "Failed to decompose matrix A." << endl;
    exit(-1);
  }

  MatrixXd x = solver.solve(GtDc);

  if(solver.info()!=Success) {
    cerr << "Failed to solve A\\b." << endl;
    exit(-1);
  }
  tsolve.toc("solving linear equations");

  // make a copy of T0
  BasicMesh Td = T0;

  // change the vertices with x
  for(int i=0;i<nverts;++i) {
    Td.set_vertex(i, x.row(i));
  }

  return Td;
}

void MeshTransferer::computeS0grad()
{
  int nfaces = S0.NumFaces();
  S0grad.resize(nfaces);
  Ds.resize(nfaces);

  for(int j=0;j<nfaces;++j) {
    // assign the reshaped gradient to the j-th row of Sgrad[i]
    auto S0ij_d = triangleGradient2(S0, j);
    S0grad[j] = S0ij_d.first;
    Ds[j] = S0ij_d.second;
  }
}

void MeshTransferer::computeT0grad()
{
  int nfaces = T0.NumFaces();
  T0grad.resize(nfaces);

  for(int j=0;j<nfaces;++j) {
    auto TV = triangleGradient(T0, j);
    auto TVinv = TV.inv();

    PhGUtils::Vector3d s;
    s.x = -TVinv(0, 0) - TVinv(1, 0);
    s.y = -TVinv(0, 1) - TVinv(1, 1);
    s.z = -TVinv(0, 2) - TVinv(1, 2);

    T0grad[j] = PhGUtils::Matrix3x3d(
                  s.x, TVinv(0, 0), TVinv(1, 0),
                  s.y, TVinv(0, 1), TVinv(1, 1),
                  s.z, TVinv(0, 2), TVinv(1, 2)
                  );
  }
}
