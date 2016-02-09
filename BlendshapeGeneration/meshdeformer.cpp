#include <MultilinearReconstruction/utils.hpp>

#include "meshdeformer.h"
#include "Geometry/geometryutils.hpp"
#include "densematrix.h"
#include "sparsematrix.h"
#include "triangle_gradient.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

MeshDeformer::MeshDeformer()
{
}


MeshDeformer::~MeshDeformer()
{
}

BasicMesh MeshDeformer::deformWithMesh(const BasicMesh &T, const PointCloud &lm_points, int itmax)
{
#define DEFORMWITHMESH_VIA_DEFORMWITHPOINTS 1
#if DEFORMWITHMESH_VIA_DEFORMWITHPOINTS
  MatrixX3d P = T.samplePoints(8, -0.1);
  return deformWithPoints(P, lm_points, itmax);
#else
  cout << "deformation with mesh ..." << endl;
  int nverts = S.verts.nrow;
  int nfaces = S.faces.nrow;

  auto &V = S.verts;
  auto &F = S.faces;

  // find the neighbor information of every vertex in the source mesh
  vector<set<int>> N(nverts);
  for (int i = 0; i < nfaces; ++i) {
    auto Fi = F.rowptr(i);
    int v1 = Fi[0], v2 = Fi[1], v3 = Fi[2];

    N[v1].insert(v2); N[v1].insert(v3);
    N[v2].insert(v1); N[v2].insert(v3);
    N[v3].insert(v1); N[v3].insert(v2);
  }

  // compute delta_i
  Array2D<double> delta(nverts, 3);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    double Si[3] = {0};
    for (auto j : Ni) {
      auto Vj = V.rowptr(j);
      Si[0] += Vj[0];
      Si[1] += Vj[1];
      Si[2] += Vj[2];
    }

    auto Vi = V.rowptr(i);

    double invNi = 1.0 / Ni.size();
    auto delta_i = delta.rowptr(i);
    delta_i[0] = Vi[0] - Si[0] * invNi;
    delta_i[1] = Vi[1] - Si[1] * invNi;
    delta_i[2] = Vi[2] - Si[2] * invNi;
  }

  auto makeVMatrix = [&](double x, double y, double z) {
    DenseMatrix V = DenseMatrix::zeros(3, 7);
    /*
     *     x   0  z -y 1 0 0
     *     y  -z  0  x 0 1 0
     *     z   y -x  0 0 0 1
     */
    V(0) =  x; V(1) =  y;  V(2) =  z;
    V(4) = -z;  V(5) =  y;
    V(6) =  z;             V(8) = -x;
    V(9) = -y; V(10) = x;
    V(12) = 1;
    V(16) = 1;
    V(20) = 1;

    return V;
  };

  // compute A matrix
  vector<DenseMatrix> A(nverts);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    A[i] = DenseMatrix::zeros(3*(Ni.size()+1), 7);

    int offset_i = i * 3;
    // set the vertex's terms
    DenseMatrix Vi = makeVMatrix(V(offset_i), V(offset_i+1), V(offset_i+2));
    // copy to A[i]
    for (int c = 0; c < 7; ++c) {
      for (int r = 0; r < 3; ++r) {
        A[i](r, c) = Vi(r, c);
      }
    }
    // set the neighbor terms
    int roffset = 3;
    for (auto j : Ni) {
      int offset_j = j * 3;
      DenseMatrix Vj = makeVMatrix(V(offset_j), V(offset_j + 1), V(offset_j + 2));

      for (int c = 0; c < 7; ++c) {
        for (int r = 0; r < 3; ++r) {
          A[i](r+roffset, c) = Vj(r, c);
        }
      }
      roffset += 3;
    }
  }

  auto makeDMatrix = [&](double x, double y, double z) {
    DenseMatrix V = DenseMatrix::zeros(3, 7);
    /*
     *     x   0  z -y 1 0 0
     *     y  -z  0  x 0 1 0
     *     z   y -x  0 0 0 1
     */
    V(0) =  x; V(1) =  y;  V(2) =  z;
    V(4) = -z;  V(5) =  y;
    V(6) =  z;             V(8) = -x;
    V(9) = -y; V(10) = x;

    return V;
  };

  // compute T matrix
  vector<DenseMatrix> Tm(nverts);
  for (int i = 0; i < nverts; ++i) {
    int offset_i = i * 3;
    DenseMatrix Di = makeDMatrix(delta(offset_i), delta(offset_i+1), delta(offset_i+2));
    auto& Ai = A[i];
    DenseMatrix At = Ai.transposed();
    DenseMatrix invAtAi = (At * Ai).inv();
    Tm[i] = Di * (invAtAi * At);
  }

  BasicMesh D = S.clone(); // make a copy of the source mesh

  // generate a point cloud from the target mesh
#if 0
  const int POINTS_PER_FACE = 2;
  int npoints = 0;
  for (int i = 0; i < T.faces.nrow; ++i) {
    // sample 8 points per face
    int fidx = i * 3;
    int v1 = T.faces(fidx), v2 = T.faces(fidx+1), v3 = T.faces(fidx+2);
    double z1 = T.verts(v1*3+2), z2 = T.verts(v2*3+2), z3 = T.verts(v3*3+2);
    double zc = (z1 + z2 + z3) / 3.0;
    if (zc > -0.1) npoints += POINTS_PER_FACE;
  }
  cout << "npoints = " << npoints << endl;
  PointCloud P;
  P.points.resize(npoints, 3);
  for (int i = 0, offset=0; i < T.faces.nrow; ++i) {
    int fidx = i * 3;
    int v1 = T.faces(fidx), v2 = T.faces(fidx+1), v3 = T.faces(fidx+2);
    double x1 = T.verts(v1*3), x2 = T.verts(v2*3), x3 = T.verts(v3*3);
    double y1 = T.verts(v1*3+1), y2 = T.verts(v2*3+1), y3 = T.verts(v3*3+1);
    double z1 = T.verts(v1*3+2), z2 = T.verts(v2*3+2), z3 = T.verts(v3*3+2);
    double zc = (z1 + z2 + z3) / 3.0;
    if (zc > -0.1) {
      for(int j=0;j<POINTS_PER_FACE;++j) {
        // sample a point
        double alpha = rand()/(double)RAND_MAX,
            beta = rand()/(double)RAND_MAX * (1-alpha),
            gamma = 1.0 - alpha - beta;

        auto Pptr = P.points.rowptr(offset); ++offset;
        Pptr[0] = x1*alpha + x2*beta + x3*gamma;
        Pptr[1] = y1*alpha + y2*beta + y3*gamma;
        Pptr[2] = z1*alpha + z2*beta + z3*gamma;
      }
    }
  }
  cout << "points sampled." << endl;
#else
  PointCloud P = T.samplePoints(16, -0.1);
  int npoints = P.points.nrow;
  ofstream fP("P.txt");
  fP << P;
  fP.close();
#endif

  // the number of matrix elements in distortion term
  int ndistortion = 0;
  for (auto Ni : N) {
    ndistortion += (Ni.size() + 1);
  }
  ndistortion *= 9;

  // main deformation loop
  const int itmax = 10;
  int iters = 0;

  double ratio_data2icp = 10.0*lm_points.points.nrow / (double) S.verts.nrow;
  cout << ratio_data2icp << endl;
  double w_icp = 0, w_icp_step = ratio_data2icp;
  double w_data = 1.0, w_data_step = 0.1;
  double w_dist = 10000.0 * ratio_data2icp, w_dist_step = 1000.0 * ratio_data2icp;
  double w_prior = 0.001, w_prior_step = 0.000095;

  while (iters++ < itmax) {
    cout << "iteration " << iters << endl;

    // find correspondence
    PhGUtils::Timer tcorr;
    tcorr.tic();
    //vector<ICPCorrespondence> icp_corr = findClosestPoints_bruteforce(P, D);
    vector<ICPCorrespondence> icp_corr = findClosestPoints_tree(P, D);
    tcorr.toc("finding correspondence");

    // compute fitting error
    double Efit = 0;
    for (int i = 0, idx=0; i < npoints; ++i) {
      // input point
      double px = P.points(idx++), py = P.points(idx++), pz = P.points(idx++);
      double dx = icp_corr[i].hit[0] - px;
      double dy = icp_corr[i].hit[1] - py;
      double dz = icp_corr[i].hit[2] - pz;

      double Ei = sqrt(dx*dx + dy*dy + dz*dz);
      Efit += Ei;
    }
    Efit /= npoints;
    cout << RED << "Efit = " << Efit << RESET << endl;

    // count the total number of terms
    int nrows = 0;
    int nterms = 0;
    // add ICP terms
    nterms += npoints * 9;
    nrows += npoints * 3;

    // add landmarks terms
    int ndata = landmarks.size();
    nterms += ndata * 3;
    nrows += ndata*3;

    // add prior terms
    int nprior = S.verts.nrow;
    nterms += nprior * 3;
    nrows += nprior * 3;

    // add distortion terms
    nterms += ndistortion;
    nrows += S.verts.nrow * 3;

    cout << "nterms = " << nterms << endl;
    cout << "nrows = " << nrows << endl;

    SparseMatrix M(nrows, nverts*3, nterms);
    vector<double> b(nrows, 0);
    int roffset = 0;

    cout << "Filling in matrix elements ..." << endl;

    // ICP term
    cout << "assembling ICP terms ..." << endl;
    PhGUtils::Timer ticp;
    ticp.tic();
    for (int i = 0; i < npoints; ++i) {
      double wi = icp_corr[i].weight * w_icp;
      int toffset = icp_corr[i].tidx*3;
      int v0 = S.faces(toffset), v1 = S.faces(toffset + 1), v2 = S.faces(toffset + 2);

      int v0offset = v0 * 3, v1offset = v1 * 3, v2offset = v2 * 3;
      double wi0 = icp_corr[i].bcoords[0] * wi;
      double wi1 = icp_corr[i].bcoords[1] * wi;
      double wi2 = icp_corr[i].bcoords[2] * wi;

      M.append(roffset, v0offset, wi0); M.append(roffset, v1offset, wi1); M.append(roffset, v2offset, wi2);
      b[roffset] = P.points(roffset) * wi;
      ++roffset;

      M.append(roffset, v0offset+1, wi0); M.append(roffset, v1offset+1, wi1); M.append(roffset, v2offset+1, wi2);
      b[roffset] = P.points(roffset) * wi;
      ++roffset;

      M.append(roffset, v0offset+2, wi0); M.append(roffset, v1offset+2, wi1); M.append(roffset, v2offset+2, wi2);
      b[roffset] = P.points(roffset) * wi;
      ++roffset;
    }
    ticp.toc("assembling ICP term");

    // landmarks term term
    cout << "assembling landmarks terms ..." << endl;
    PhGUtils::Timer tland;
    tland.tic();
    for (int i = 0, ioffset=0; i < ndata; ++i) {
      int dstart = landmarks[i] * 3;
      double wi = w_data;

      /*
      cout << "("
           << lm_points.points(ioffset) << ","
           << lm_points.points(ioffset+1) << ","
           << lm_points.points(ioffset+2)
           << ")"
           << " vs "
           << "("
           << V(dstart) << ","
           << V(dstart+1) << ","
           << V(dstart+2)
           << ") " << wi
           << endl;
      */

      M.append(roffset, dstart, wi); ++dstart;
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, dstart, wi); ++dstart;
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, dstart, wi);
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;
    }
    tland.toc("assembling landmarks term");

    // prior term, i.e. similarity to source mesh
    cout << "assembling prior terms ..." << endl;
    PhGUtils::Timer tprior;
    tprior.tic();
    for (int i = 0, ioffset = 0; i < nprior; ++i) {
      double wi = w_prior;
      M.append(roffset, ioffset, wi);
      b[roffset] = S.verts(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, ioffset, wi);
      b[roffset] = S.verts(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, ioffset, wi);
      b[roffset] = S.verts(ioffset) * wi;
      ++roffset; ++ioffset;
    }
    tprior.toc("assembling prior term");

    // Laplacian distortion term
    PhGUtils::Timer tdist;
    tdist.tic();
    cout << "assembling Laplacian terms ..." << endl;
    for (int i = 0; i < nverts; ++i) {
      auto& Ti = Tm[i];
      auto& Ni = N[i];
      double wi = w_dist;

      int istart = i * 3;
      // deformation part
      //M.append(roffset+istart+0, istart+0, wi);
      //M.append(roffset+istart+1, istart+1, wi);
      //M.append(roffset+istart+2, istart+2, wi);

      M.append(roffset+istart+0, istart+0, (1 - Ti(0, 0))*wi);
      M.append(roffset+istart+0, istart+1, (0 - Ti(0, 1))*wi);
      M.append(roffset+istart+0, istart+2, (0 - Ti(0, 2))*wi);

      M.append(roffset+istart+1, istart+0, (0 - Ti(1, 0))*wi);
      M.append(roffset+istart+1, istart+1, (1 - Ti(1, 1))*wi);
      M.append(roffset+istart+1, istart+2, (0 - Ti(1, 2))*wi);

      M.append(roffset+istart+2, istart+0, (0 - Ti(2, 0))*wi);
      M.append(roffset+istart+2, istart+1, (0 - Ti(2, 1))*wi);
      M.append(roffset+istart+2, istart+2, (1 - Ti(2, 2))*wi);

      int j = 1;
      double wij = -1.0 / Ni.size();
      for (auto Nij : Ni) {
        int jstart = Nij * 3;
        int joffset = j * 3; ++j;
        //M.append(roffset+istart+0, jstart+0, wij*wi);
        //M.append(roffset+istart+1, jstart+1, wij*wi);
        //M.append(roffset+istart+2, jstart+2, wij*wi);

        M.append(roffset+istart+0, jstart+0, (wij - Ti(0, joffset+0))*wi);
        M.append(roffset+istart+0, jstart+1, (  0 - Ti(0, joffset+1))*wi);
        M.append(roffset+istart+0, jstart+2, (  0 - Ti(0, joffset+2))*wi);

        M.append(roffset+istart+1, jstart+0, (  0 - Ti(1, joffset+0))*wi);
        M.append(roffset+istart+1, jstart+1, (wij - Ti(1, joffset+1))*wi);
        M.append(roffset+istart+1, jstart+2, (  0 - Ti(1, joffset+2))*wi);

        M.append(roffset+istart+2, jstart+0, (  0 - Ti(2, joffset+0))*wi);
        M.append(roffset+istart+2, jstart+1, (  0 - Ti(2, joffset+1))*wi);
        M.append(roffset+istart+2, jstart+2, (wij - Ti(2, joffset+2))*wi);
      }
    }
    tdist.toc("assembling distortion term");

    cout << nterms << endl;

    // solve sparse linear system
    cout << "M matrix assembled..." << endl;
    // compute M' * M
    cout << "computing M'*M..." << endl;

    ofstream mfout("M.txt");
    mfout << M << endl;
    mfout.close();

    PhGUtils::Timer tsolve;
    tsolve.tic();

    auto Ms = M.to_sparse();
    auto Mt = cholmod_transpose(Ms, 2, global::cm);
    auto MtM = cholmod_aat(Mt, NULL, 0, 1, global::cm);
    MtM->stype = 1;

    // compute M' * b
    cout << "computing M'*b..." << endl;
    cholmod_dense *bs = cholmod_allocate_dense(Ms->nrow, 1, Ms->nrow, CHOLMOD_REAL, global::cm);
    memcpy(bs->x, &(b[0]), sizeof(double)*Ms->nrow);
    cholmod_dense *Mtb = cholmod_allocate_dense(MtM->nrow, 1, MtM->nrow, CHOLMOD_REAL, global::cm);
    double alpha[2] = {1, 0}; double beta[2] = {0, 0};

    //cholmod_print_sparse(Mt, "Mt", global::cm);
    //cholmod_print_dense(bs, "bs", global::cm);
    //cholmod_print_dense(Mtb, "Mtb", global::cm);
    cholmod_sdmult(Mt, 0, alpha, beta, bs, Mtb, global::cm);

    cout << "Solving (M'*M)\(M'*b) ..." << endl;
    // solve (M'*M)\(M'*b)
    // solution vector
    cholmod_factor *L = cholmod_analyze(MtM, global::cm);
    cholmod_factorize(MtM, L, global::cm);
    cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, Mtb, global::cm);
    cout << "done." << endl;
    tsolve.toc("solving linear equations");

    //cholmod_print_dense(x, "x", global::cm);
    ofstream fout("x.txt");
    for(int xidx=0;xidx<x->nrow;++xidx) fout << ((double*)x->x) [xidx] << endl;
    fout.close();

    // update the vertices of D using the x vector
    memcpy(D.verts.data.get(), x->x, sizeof(double)*nverts*3);

    // release memory

    cholmod_free_sparse(&Ms, global::cm);
    cholmod_free_sparse(&Mt, global::cm);
    cholmod_free_sparse(&MtM, global::cm);
    cholmod_free_dense(&bs, global::cm);
    cholmod_free_dense(&Mtb, global::cm);
    cholmod_free_factor(&L, global::cm);
    cholmod_free_dense(&x, global::cm);


    // update weighting factors
    // increase w_icp
    w_icp = iters * w_icp_step;
    // decrease w_data
    //      w_data = (itmax - iters) * w_data_step;
    // decrease w_dist
    w_dist = sqrt((itmax - iters) * w_dist_step);
    // decrease w_prior
    w_prior = (itmax - iters) * w_prior_step;
  }

  return D;
#endif
}

BasicMesh MeshDeformer::deformWithPoints(const MatrixX3d &P, const PointCloud &lm_points, int itmax)
{
  //cout << "deformation with mesh ..." << endl;
  int nverts = S.NumVertices();
  int nfaces = S.NumFaces();

  // find the neighbor information of every vertex in the source mesh
  vector<set<int>> N(nverts);
  for (int i = 0; i < nfaces; ++i) {
    auto Fi = S.face(i);
    int v1 = Fi[0], v2 = Fi[1], v3 = Fi[2];

    N[v1].insert(v2); N[v1].insert(v3);
    N[v2].insert(v1); N[v2].insert(v3);
    N[v3].insert(v1); N[v3].insert(v2);
  }

  // compute delta_i
  Array2D<double> delta(nverts, 3);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    double Si[3] = {0};
    for (auto j : Ni) {
      auto Vj = S.vertex(j);
      Si[0] += Vj[0];
      Si[1] += Vj[1];
      Si[2] += Vj[2];
    }

    auto Vi = S.vertex(i);

    double invNi = 1.0 / Ni.size();
    auto delta_i = delta.rowptr(i);
    delta_i[0] = Vi[0] - Si[0] * invNi;
    delta_i[1] = Vi[1] - Si[1] * invNi;
    delta_i[2] = Vi[2] - Si[2] * invNi;
  }

  auto makeVMatrix = [&](double x, double y, double z) {
    DenseMatrix V = DenseMatrix::zeros(3, 7);
    /*
     *     x   0  z -y 1 0 0
     *     y  -z  0  x 0 1 0
     *     z   y -x  0 0 0 1
     */
    V(0) =  x; V(1) =  y;  V(2) =  z;
    V(4) = -z;  V(5) =  y;
    V(6) =  z;             V(8) = -x;
    V(9) = -y; V(10) = x;
    V(12) = 1;
    V(16) = 1;
    V(20) = 1;

    return V;
  };

  // compute A matrix
  vector<DenseMatrix> A(nverts);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    A[i] = DenseMatrix::zeros(3*(Ni.size()+1), 7);

    int offset_i = i * 3;
    // set the vertex's terms
    auto vert_i = S.vertex(i);
    DenseMatrix Vi = makeVMatrix(vert_i[0], vert_i[1], vert_i[2]);

    // copy to A[i]
    for (int c = 0; c < 7; ++c) {
      for (int r = 0; r < 3; ++r) {
        A[i](r, c) = Vi(r, c);
      }
    }

    // set the neighbor terms
    int roffset = 3;
    for (auto j : Ni) {
      int offset_j = j * 3;
      auto vert_j = S.vertex(j);
      DenseMatrix Vj = makeVMatrix(vert_j[0], vert_j[1], vert_j[2]);

      for (int c = 0; c < 7; ++c) {
        for (int r = 0; r < 3; ++r) {
          A[i](r+roffset, c) = Vj(r, c);
        }
      }
      roffset += 3;
    }
  }

  auto makeDMatrix = [&](double x, double y, double z) {
    DenseMatrix V = DenseMatrix::zeros(3, 7);
    /*
     *     x   0  z -y 1 0 0
     *     y  -z  0  x 0 1 0
     *     z   y -x  0 0 0 1
     */
    V(0) =  x; V(1) =  y;  V(2) =  z;
    V(4) = -z; V(5) =  y;
    V(6) =  z;             V(8) = -x;
    V(9) = -y; V(10) = x;

    return V;
  };

  // compute T matrix
  vector<DenseMatrix> Tm(nverts);
  for (int i = 0; i < nverts; ++i) {
    int offset_i = i * 3;
    DenseMatrix Di = makeDMatrix(delta(offset_i), delta(offset_i+1), delta(offset_i+2));
    auto& Ai = A[i];
    DenseMatrix At = Ai.transposed();
    DenseMatrix AtAi = At * Ai;
    //cout << AtAi << endl;
    DenseMatrix invAtAi = AtAi.inv();
    //cout << invAtAi << endl;
    Tm[i] = Di * (invAtAi * At);
    //cout << Tm[i] << endl;
  }

  BasicMesh D = S; // make a copy of the source mesh

  int npoints = P.rows();
  ofstream fP("P.txt");
  fP << P;
  fP.close();

  // the number of matrix elements in distortion term
  int ndistortion = 0;
  for (auto Ni : N) {
    ndistortion += (Ni.size() + 1);
  }
  ndistortion *= 9;

  // main deformation loop
  int iters = 0;

  double ratio_data2icp = max(0.001, 10.0*lm_points.points.nrow / (double) S.NumVertices());
  //cout << ratio_data2icp << endl;
  double w_icp = 0, w_icp_step = ratio_data2icp;
  double w_data = 10.0, w_data_step = w_data/itmax;
  double w_dist = 10000.0 * ratio_data2icp, w_dist_step = w_dist/itmax;
  double w_prior = 0.001, w_prior_step = w_prior*0.95/itmax;

#define ANALYZE_ONCE 1  // analyze MtM only once
#if ANALYZE_ONCE
  cholmod_factor *L = nullptr;
#endif
  while (iters++ < itmax) {
    cout << "iteration " << iters << endl;

    // find correspondence
    PhGUtils::Timer tcorr;
    tcorr.tic();
    //vector<ICPCorrespondence> icp_corr = findClosestPoints_bruteforce(P, D);
    vector<ICPCorrespondence> icp_corr = findClosestPoints_tree(P, D);
    tcorr.toc("finding correspondence");

    // compute fitting error
    double Efit = 0;
    for (int i = 0, idx=0; i < npoints; ++i) {
      // input point
      double px = P(i, 0), py = P(i, 1), pz = P(i, 2);
      double dx = icp_corr[i].hit[0] - px;
      double dy = icp_corr[i].hit[1] - py;
      double dz = icp_corr[i].hit[2] - pz;

      double Ei = sqrt(dx*dx + dy*dy + dz*dz);
      Efit += Ei;
      //Efit = max(Efit, Ei);
    }
    Efit /= npoints;
    ColorStream(ColorOutput::Red)<< "Efit = " << Efit;

    // count the total number of terms
    int nrows = 0;
    int nterms = 0;
    // add ICP terms
    nterms += npoints * 9;
    nrows += npoints * 3;

    // add landmarks terms
    int ndata = landmarks.size();
    nterms += ndata * 3;
    nrows += ndata * 3;

    // add prior terms
    int nprior = S.NumVertices();
    nterms += nprior * 3;
    nrows += nprior * 3;

    // add distortion terms
    nterms += ndistortion;
    nrows += S.NumVertices() * 3;

    //cout << "nterms = " << nterms << endl;
    //cout << "nrows = " << nrows << endl;

    SparseMatrix M(nrows, nverts*3, nterms);
    vector<double> b(nrows, 0);
    int roffset = 0;

    //cout << "Filling in matrix elements ..." << endl;

    // ICP term
    //cout << "assembling ICP terms ..." << endl;
    PhGUtils::Timer ticp;
    ticp.tic();
    for (int i = 0; i < npoints; ++i) {
      double wi = icp_corr[i].weight * w_icp;
      int toffset = icp_corr[i].tidx*3;
      auto face_t = S.face(icp_corr[i].tidx);
      int v0 = face_t[0], v1 = face_t[1], v2 = face_t[2];

      int v0offset = v0 * 3, v1offset = v1 * 3, v2offset = v2 * 3;
      double wi0 = icp_corr[i].bcoords[0] * wi;
      double wi1 = icp_corr[i].bcoords[1] * wi;
      double wi2 = icp_corr[i].bcoords[2] * wi;

      M.append(roffset, v0offset, wi0); M.append(roffset, v1offset, wi1); M.append(roffset, v2offset, wi2);
      b[roffset] = P(i, 0) * wi;
      ++roffset;

      M.append(roffset, v0offset+1, wi0); M.append(roffset, v1offset+1, wi1); M.append(roffset, v2offset+1, wi2);
      b[roffset] = P(i, 1) * wi;
      ++roffset;

      M.append(roffset, v0offset+2, wi0); M.append(roffset, v1offset+2, wi1); M.append(roffset, v2offset+2, wi2);
      b[roffset] = P(i, 2) * wi;
      ++roffset;
    }
    ticp.toc("assembling ICP term");

    // landmarks term term
    //cout << "assembling landmarks terms ..." << endl;
    PhGUtils::Timer tland;
    tland.tic();
    for (int i = 0, ioffset=0; i < ndata; ++i) {
      int dstart = landmarks[i] * 3;
      double wi = w_data;

      /*
      cout << "("
           << lm_points.points(ioffset) << ","
           << lm_points.points(ioffset+1) << ","
           << lm_points.points(ioffset+2)
           << ")"
           << " vs "
           << "("
           << V(dstart) << ","
           << V(dstart+1) << ","
           << V(dstart+2)
           << ") " << wi
           << endl;
      */

      M.append(roffset, dstart, wi); ++dstart;
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, dstart, wi); ++dstart;
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;

      M.append(roffset, dstart, wi);
      b[roffset] = lm_points.points(ioffset) * wi;
      ++roffset; ++ioffset;
    }
    tland.toc("assembling landmarks term");

    // prior term, i.e. similarity to source mesh
    //cout << "assembling prior terms ..." << endl;
    PhGUtils::Timer tprior;
    tprior.tic();
    for (int i = 0, ioffset = 0; i < nprior; ++i) {
      double wi = w_prior;
      auto vert_i = S.vertex(i);
      M.append(roffset, ioffset, wi);
      b[roffset] = vert_i[0] * wi;
      ++roffset; ++ioffset;

      M.append(roffset, ioffset, wi);
      b[roffset] = vert_i[1] * wi;
      ++roffset; ++ioffset;

      M.append(roffset, ioffset, wi);
      b[roffset] = vert_i[2] * wi;
      ++roffset; ++ioffset;
    }
    tprior.toc("assembling prior term");

    // Laplacian distortion term
    PhGUtils::Timer tdist;
    tdist.tic();
    //cout << "assembling Laplacian terms ..." << endl;
    for (int i = 0; i < nverts; ++i) {
      auto& Ti = Tm[i];
      auto& Ni = N[i];
      double wi = w_dist;

      //cout << Ti << endl;

      int istart = i * 3;
      // deformation part
      //M.append(roffset+istart+0, istart+0, wi);
      //M.append(roffset+istart+1, istart+1, wi);
      //M.append(roffset+istart+2, istart+2, wi);

      M.append(roffset+istart+0, istart+0, (1 - Ti(0, 0))*wi);
      M.append(roffset+istart+0, istart+1, (0 - Ti(0, 1))*wi);
      M.append(roffset+istart+0, istart+2, (0 - Ti(0, 2))*wi);

      M.append(roffset+istart+1, istart+0, (0 - Ti(1, 0))*wi);
      M.append(roffset+istart+1, istart+1, (1 - Ti(1, 1))*wi);
      M.append(roffset+istart+1, istart+2, (0 - Ti(1, 2))*wi);

      M.append(roffset+istart+2, istart+0, (0 - Ti(2, 0))*wi);
      M.append(roffset+istart+2, istart+1, (0 - Ti(2, 1))*wi);
      M.append(roffset+istart+2, istart+2, (1 - Ti(2, 2))*wi);

      int j = 1;
      double wij = -1.0 / Ni.size();
      for (auto Nij : Ni) {
        int jstart = Nij * 3;
        int joffset = j * 3; ++j;
        //M.append(roffset+istart+0, jstart+0, wij*wi);
        //M.append(roffset+istart+1, jstart+1, wij*wi);
        //M.append(roffset+istart+2, jstart+2, wij*wi);

        M.append(roffset+istart+0, jstart+0, (wij - Ti(0, joffset+0))*wi);
        M.append(roffset+istart+0, jstart+1, (  0 - Ti(0, joffset+1))*wi);
        M.append(roffset+istart+0, jstart+2, (  0 - Ti(0, joffset+2))*wi);

        M.append(roffset+istart+1, jstart+0, (  0 - Ti(1, joffset+0))*wi);
        M.append(roffset+istart+1, jstart+1, (wij - Ti(1, joffset+1))*wi);
        M.append(roffset+istart+1, jstart+2, (  0 - Ti(1, joffset+2))*wi);

        M.append(roffset+istart+2, jstart+0, (  0 - Ti(2, joffset+0))*wi);
        M.append(roffset+istart+2, jstart+1, (  0 - Ti(2, joffset+1))*wi);
        M.append(roffset+istart+2, jstart+2, (wij - Ti(2, joffset+2))*wi);
      }
    }
    tdist.toc("assembling distortion term");

    //cout << nterms << endl;

    // solve sparse linear system
    //cout << "M matrix assembled..." << endl;
    // compute M' * M
    //cout << "computing M'*M..." << endl;


//    ofstream mfout("M.txt");
//    mfout << M << endl;
//    mfout.close();


    PhGUtils::Timer tmatrix;
    tmatrix.tic();

    auto Ms = M.to_sparse();
    auto Mt = cholmod_transpose(Ms, 2, global::cm);
    auto MtM = cholmod_aat(Mt, NULL, 0, 1, global::cm);
    MtM->stype = 1;

    // compute M' * b
    //cout << "computing M'*b..." << endl;
    cholmod_dense *bs = cholmod_allocate_dense(Ms->nrow, 1, Ms->nrow, CHOLMOD_REAL, global::cm);
    memcpy(bs->x, &(b[0]), sizeof(double)*Ms->nrow);
    cholmod_dense *Mtb = cholmod_allocate_dense(MtM->nrow, 1, MtM->nrow, CHOLMOD_REAL, global::cm);
    tmatrix.toc("constructing linear equations");

    PhGUtils::Timer tsolve;
    tsolve.tic();

    double alpha[2] = {1, 0}; double beta[2] = {0, 0};

    //cholmod_print_sparse(Mt, "Mt", global::cm);
    //cholmod_print_dense(bs, "bs", global::cm);
    //cholmod_print_dense(Mtb, "Mtb", global::cm);
    cholmod_sdmult(Mt, 0, alpha, beta, bs, Mtb, global::cm);

    //cout << "Solving (M'*M)\(M'*b) ..." << endl;
    // solve (M'*M)\(M'*b)
    // solution vector
#if ANALYZE_ONCE
    if( L == nullptr )
      L = cholmod_analyze(MtM, global::cm);
#else
    cholmod_factor *L = cholmod_analyze(MtM, global::cm);
#endif

    cholmod_factorize(MtM, L, global::cm);
    cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, Mtb, global::cm);
    //cout << "done." << endl;
    tsolve.toc("solving linear equations");

    //cholmod_print_dense(x, "x", global::cm);

    /*
    ofstream fout("x.txt");
    for(int xidx=0;xidx<x->nrow;++xidx) fout << ((double*)x->x) [xidx] << endl;
    fout.close();
    */

    // update the vertices of D using the x vector
#if 0
    memcpy(D.verts.data.get(), x->x, sizeof(double)*nverts*3);
#else
    const double* vertex_ptr = static_cast<double*>(x->x);
    for(int i=0;i<nverts;++i) {
      D.set_vertex(i, Vector3d(vertex_ptr[i*3+0],
                               vertex_ptr[i*3+1],
                               vertex_ptr[i*3+2]));
    }
#endif

    // release memory
    cholmod_free_sparse(&Ms, global::cm);
    cholmod_free_sparse(&Mt, global::cm);
    cholmod_free_sparse(&MtM, global::cm);
    cholmod_free_dense(&bs, global::cm);
    cholmod_free_dense(&Mtb, global::cm);
#if !ANALYZE_ONCE
    cholmod_free_factor(&L, global::cm);
#endif
    cholmod_free_dense(&x, global::cm);

    // update weighting factors
    // increase w_icp
    w_icp = iters * w_icp_step;
    // decrease w_data
    //      w_data = (itmax - iters) * w_data_step;
    // decrease w_dist
    w_dist = sqrt((itmax - iters) * w_dist_step);
    // decrease w_prior
    w_prior = (itmax - iters) * w_prior_step;
  }
#if ANALYZE_ONCE
  cholmod_free_factor(&L, global::cm);
#endif
  return D;
}

vector<ICPCorrespondence> MeshDeformer::findClosestPoints_tree(const MatrixX3d &P, const BasicMesh &mesh) {
  int nfaces = mesh.NumFaces();
  int npoints = P.rows();

  std::vector<Triangle> triangles;
  triangles.reserve(nfaces);
  for(int i=0,ioffset=0;i<nfaces;++i) {
    auto face_i = mesh.face(i);
    int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
    auto p1 = mesh.vertex(v1), p2 = mesh.vertex(v2), p3 = mesh.vertex(v3);
    Point a(p1[0], p1[1], p1[2]);
    Point b(p2[0], p2[1], p2[2]);
    Point c(p3[0], p3[1], p3[2]);

    triangles.push_back(Triangle(a, b, c));
  }

  Tree tree(triangles.begin(), triangles.end());

  vector<ICPCorrespondence> corrs(npoints);

  // query the tree for closest point
#pragma omp parallel for
  for (int pidx = 0; pidx < npoints; ++pidx) {
    int poffset = pidx * 3;
    double px = P(pidx, 0), py = P(pidx, 1), pz = P(pidx, 2);

#undef max
    ICPCorrespondence bestCorr;
    bestCorr.d = numeric_limits<double>::max();
    Tree::Point_and_primitive_id bestHit = tree.closest_point_and_primitive(Point(px, py, pz));
    bestCorr.tidx = bestHit.second - triangles.begin();
    bestCorr.hit[0] = bestHit.first.x(); bestCorr.hit[1] = bestHit.first.y(); bestCorr.hit[2] = bestHit.first.z();
    double dx = px - bestHit.first.x(), dy = py - bestHit.first.y(), dz = pz - bestHit.first.z();
    bestCorr.d = dx*dx+dy+dy+dz*dz;

    // compute bary-centric coordinates
    int toffset = bestCorr.tidx * 3;
    auto face_t = mesh.face(bestCorr.tidx);
    int v0idx = face_t[0], v1idx = face_t[1], v2idx = face_t[2];

    auto v0 = mesh.vertex(v0idx);
    auto v1 = mesh.vertex(v1idx);
    auto v2 = mesh.vertex(v2idx);

    PhGUtils::Point3f bcoords;
    PhGUtils::computeBarycentricCoordinates(PhGUtils::Point3f(px, py, pz),
                                            PhGUtils::Point3f(v0[0], v0[1], v0[2]),
                                            PhGUtils::Point3f(v1[0], v1[1], v1[2]),
                                            PhGUtils::Point3f(v2[0], v2[1], v2[2]),
                                            bcoords);
    bestCorr.bcoords[0] = bcoords.x; bestCorr.bcoords[1] = bcoords.y; bestCorr.bcoords[2] = bcoords.z;

    // compute face normal
    PhGUtils::Vector3d p0(v0[0], v0[1], v0[2]),
                       p1(v1[0], v1[1], v1[2]),
                       p2(v2[0], v2[1], v2[2]);
    PhGUtils::Vector3d normal(p1-p0, p2-p0);
    PhGUtils::Vector3d dvec(dx, dy, dz);
    bestCorr.weight = normal.normalized().dot(dvec.normalized());

    corrs[pidx] = bestCorr;
  }
  return corrs;
}

vector<ICPCorrespondence> MeshDeformer::findClosestPoints_bruteforce(const MatrixX3d &P, const BasicMesh &mesh)
{
  int nfaces = mesh.NumFaces();
  int npoints = P.rows();
  vector<ICPCorrespondence> corrs(npoints);
#pragma omp parallel for
  for (int pidx = 0; pidx < npoints; ++pidx) {
    int poffset = pidx * 3;
    double px = P(pidx, 0), py = P(pidx, 1), pz = P(pidx, 2);

#undef max
    ICPCorrespondence bestCorr;
    bestCorr.d = numeric_limits<double>::max();

    for (int i = 0, foffset=0; i < nfaces; ++i, foffset+=3) {
      auto face_i = mesh.face(i);
      int v0 = face_i[0], v1 = face_i[1], v2 = face_i[2];
      auto p0 = mesh.vertex(v0), p1 = mesh.vertex(v1), p2 = mesh.vertex(v2);
      // find closest point on triangle
      ICPCorrespondence corr = findClosestPoint_triangle(px, py, pz,
                                                         p0, p1, p2);
      corr.tidx = i;
      if (corr.d < bestCorr.d) bestCorr = corr;
    }
    // compute bary-centric coordinates
    int toffset = bestCorr.tidx * 3;
    auto face_t = mesh.face(bestCorr.tidx);
    int v0idx = face_t[0], v1idx = face_t[1], v2idx = face_t[2];

    auto v0 = mesh.vertex(v0idx);
    auto v1 = mesh.vertex(v1idx);
    auto v2 = mesh.vertex(v2idx);

    PhGUtils::Point3f bcoords;
    PhGUtils::computeBarycentricCoordinates(PhGUtils::Point3f(px, py, pz),
                                            PhGUtils::Point3f(v0[0], v0[1], v0[2]),
                                            PhGUtils::Point3f(v1[0], v1[1], v1[2]),
                                            PhGUtils::Point3f(v2[0], v2[1], v2[2]),
                                            bcoords);
    bestCorr.bcoords[0] = bcoords.x; bestCorr.bcoords[1] = bcoords.y; bestCorr.bcoords[2] = bcoords.z;
    bestCorr.weight = 1.0;

    corrs[pidx] = bestCorr;
  }
  return corrs;
}

ICPCorrespondence MeshDeformer::findClosestPoint_triangle(double px, double py, double pz, const Vector3d& v0, const Vector3d& v1, const Vector3d& v2)
{
  ICPCorrespondence corr;
  PhGUtils::Point3d hit;
  corr.d = PhGUtils::pointToTriangleDistance(
    PhGUtils::Point3d(px, py, pz),
    PhGUtils::Point3d(v0[0], v0[1], v0[2]),
    PhGUtils::Point3d(v1[0], v1[1], v1[2]),
    PhGUtils::Point3d(v2[0], v2[1], v2[2]),
    hit
  );
  corr.hit[0] = hit.x; corr.hit[1] = hit.y; corr.hit[2] = hit.z;
  return corr;
}
