#include "meshtransferer.h"
#include "Geometry/matrix.hpp"
#include "utils.h"
#include "sparsematrix.h"
#include "densematrix.h"

MeshTransferer::MeshTransferer()
{

}

MeshTransferer::~MeshTransferer()
{

}

BasicMesh MeshTransferer::transfer(const BasicMesh &S1)
{
  // compute the gradients of S0 and T0
  int nfaces = S0.faces.nrow;
  int nverts = S0.verts.nrow;

  // use vertex formulation to solve it
  vector<PhGUtils::Matrix3x3d> S(nfaces);
  vector<PhGUtils::Matrix3x3d> T(nfaces);
  Array1D<double> Ds(nfaces);
  for(int j=0;j<nfaces;++j) {
    // assign the reshaped gradient to the j-th row of Sgrad[i]
    auto S0ij_d = triangleGradient2(S0, j);
    auto V = S0ij_d.first;
    Ds(j) = S0ij_d.second;

    auto Vt = triangleGradient(S1, j);

    S[j] = (Vt * V.inv()).transposed();

    auto TV = triangleGradient(T0, j);
    auto TVinv = TV.inv();

    PhGUtils::Vector3d s;
    s.x = -TVinv(0, 0) - TVinv(1, 0);
    s.y = -TVinv(0, 1) - TVinv(1, 1);
    s.z = -TVinv(0, 2) - TVinv(1, 2);

    T[j] = PhGUtils::Matrix3x3d(
             s.x, TVinv(0, 0), TVinv(1, 0),
             s.y, TVinv(0, 1), TVinv(1, 1),
             s.z, TVinv(0, 2), TVinv(1, 2)
             );
  }

  // assemble sparse matrix A
  int nrowsA = nfaces * 3;
  int nsv = stationary_vertices.size();
  int nrowsC = nsv;
  int nrows = nrowsA + nrowsC;
  int ncols = nverts;
  int ntermsA = nfaces*9;
  int ntermsC = stationary_vertices.size();
  int nterms = ntermsA + ntermsC;
  SparseMatrix A(nrows, ncols, nterms);
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
    int *f = S0.faces.rowptr(i);

    auto Ti = T[i];

    A.append(ioffset, f[0], Ti(0));
    A.append(ioffset, f[1], Ti(1));
    A.append(ioffset, f[2], Ti(2));
    ++ioffset;

    A.append(ioffset, f[0], Ti(3));
    A.append(ioffset, f[1], Ti(4));
    A.append(ioffset, f[2], Ti(5));
    ++ioffset;

    A.append(ioffset, f[0], Ti(6));
    A.append(ioffset, f[1], Ti(7));
    A.append(ioffset, f[2], Ti(8));
    ++ioffset;
  }

  // fill in the lower part of A, stationary vertices part
  for(int i=0;i<nsv;++i) {
    A.append(nrowsA+i, stationary_vertices[i], 1);
  }

  ofstream fA("A.txt");
  fA<<A;
  fA.close();

  // fill in c matrix
  DenseMatrix c(nrows, 3);
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
      auto vj = T0.verts.rowptr(stationary_vertices[j]);
      c(joffset, i) = vj[i];
    }
  }

  cholmod_sparse *G = A.to_sparse();
  cholmod_sparse *Gt = cholmod_transpose(G, 2, global::cm);

  // compute GtD
  // just multiply Dsi to corresponding elemenets
  double *Gtx = (double*)Gt->x;
  const int* Gtp = (const int*)(Gt->p);
  for(int i=0;i<nrowsA;++i) {
    int fidx = i/3;
    for(int j=Gtp[i];j<Gtp[i+1];++j) {
      Gtx[j] *= Ds(fidx);
    }
  }

  // compute GtDG
  cholmod_sparse *GtDG = cholmod_ssmult(Gt, G, 0, 1, 1, global::cm);
  GtDG->stype = 1;

  // compute GtD * c
  cholmod_dense *GtDc = cholmod_allocate_dense(ncols, 3, ncols, CHOLMOD_REAL, global::cm);
  double alpha[2] = {1, 0}; double beta[2] = {0, 0};
  cholmod_sdmult(Gt, 0, alpha, beta, c.to_dense(), GtDc, global::cm);

  // solve for GtDG \ GtDc
  cholmod_factor *L = cholmod_analyze(GtDG, global::cm);
  cholmod_factorize(GtDG, L, global::cm);
  cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, GtDc, global::cm);

  // make a copy of T0
  BasicMesh Td = T0;

  // change the vertices with x
  double *Vx = (double*)x->x;
  for(int i=0;i<nverts;++i) {
    Td.verts(i, 0) = Vx[i];
    Td.verts(i, 1) = Vx[i+nverts];
    Td.verts(i, 2) = Vx[i+nverts*2];
  }

  // release memory
  cholmod_free_sparse(&G, global::cm);
  cholmod_free_sparse(&Gt, global::cm);
  cholmod_free_sparse(&GtDG, global::cm);
  cholmod_free_dense(&GtDc, global::cm);
  cholmod_free_factor(&L, global::cm);
  cholmod_free_dense(&x, global::cm);

  return Td;
}

