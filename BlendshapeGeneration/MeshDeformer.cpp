#include "MeshDeformer.h"
#include "Geometry/geometryutils.hpp"
#include "densematrix.h"
#include "sparsematrix.h"

MeshDeformer::MeshDeformer()
{
}


MeshDeformer::~MeshDeformer()
{
}

BasicMesh MeshDeformer::deformWithMesh(const BasicMesh &T, const PointCloud &lm_points)
{
  cout << "deformation with mesh ..." << endl;
  int nverts = S.verts.nrow;
  int nfaces = S.faces.nrow;

  auto &V = S.verts;
  auto &F = S.faces;

  // find the neighbor information of every vertex in the source mesh
  vector<unordered_set<int>> N(nverts);
  for (int i = 0,offset=0; i < nfaces; ++i) {
    int v1 = F(offset); ++offset;
    int v2 = F(offset); ++offset;
    int v3 = F(offset); ++offset;

    N[v1].insert(v2); N[v1].insert(v3);
    N[v2].insert(v1); N[v1].insert(v3);
    N[v3].insert(v1); N[v1].insert(v2);
  }

  // compute delta_i
  Array2D<double> delta(nverts, 3);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    double sx = 0, sy = 0, sz = 0;
    for (auto j : Ni) {
      int offset_j = j * 3;
      sx += V(offset_j); ++offset_j;
      sy += V(offset_j); ++offset_j;
      sz += V(offset_j); ++offset_j;
    }

    int offset_i = i * 3;
    double invNi = 1.0 / Ni.size();
    delta(offset_i  ) = V(offset_i  ) - sx * invNi;
    delta(offset_i+1) = V(offset_i+1) - sy * invNi;
    delta(offset_i+2) = V(offset_i+2) - sz * invNi;
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
    auto Vi = makeVMatrix(V(offset_i), V(offset_i+1), V(offset_i+2));
    // copy to A[i]
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 7; ++c) {
        A[i](r * 7 + c) = Vi(r * 7 + c);
      }
    }
    // set the neighbor terms
    int roffset = 3;
    for (auto j : Ni) {
      int offset_j = j * 3;
      auto Vj = makeVMatrix(V(offset_j), V(offset_j + 1), V(offset_j + 2));

      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 7; ++c) {
          A[i]((r+roffset) * 7 + c) = Vj(r * 7 + c);
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
    auto Di = makeDMatrix(delta(offset_i), delta(offset_i+1), delta(offset_i+2));
    auto& Ai = A[i];
    auto At = A[i].transposed();
    auto invAtAi = (At * Ai).inv();
    Tm[i] = Di * (invAtAi * At);
  }

  BasicMesh D = S.clone(); // make a copy of the source mesh

  // generate a point cloud from the target mesh  
  int npoints = 0;
  for (int i = 0; i < T.verts.nrow; ++i) {
    if (T.verts(i * 3 + 2) > -0.1) ++npoints;
  }
  PointCloud P;
  P.points.resize(npoints, 3);
  for (int i = 0, ridx=0; i < T.verts.nrow; ++i) {
    int offset_i = i * 3;
    if (T.verts(offset_i + 2) > -0.1) {
      int offset = ridx * 3;      
      P.points(offset) = T.verts(offset_i); ++offset; ++offset_i;
      P.points(offset) = T.verts(offset_i); ++offset; ++offset_i;
      P.points(offset) = T.verts(offset_i);
      ++ridx;
    }
  }

  // the number of matrix elements in distortion term
  int ndistortion = 0;
  for (auto Ni : N) {
    ndistortion += (Ni.size() + 1);
  }
  ndistortion *= 9;

  // main deformation loop
  const int itmax = 1;
  int iters = 0;

  double w_icp = 1e-6, w_icp_step = landmarks.size() / (double)npoints;
  double w_data = 1.0, w_data_step = 0.1;
  double w_dist = 100.0, w_dist_step = 10.0;
  double w_prior = 0.1, w_prior_step = 0.0095;

  cholmod_common common;
  cholmod_start(&common);

  while (iters++ < itmax) {
    cout << "iteration " << iters << endl;

    // find correspondence
    vector<ICPCorrespondence> icp_corr = findClosestPoints_bruteforce(P, D);

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
    cout << "Efit = " << Efit << endl;
    
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
    vector<double> b(nterms, 0);
    int roffset = 0;

    cout << "Filling in matrix elements ..." << endl;

    // ICP term
    cout << "ICP terms ..." << endl;
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

    // landmarks term term
    cout << "landmarks terms ..." << endl;
    for (int i = 0, ioffset=0; i < ndata; ++i) {
      int dstart = landmarks[i] * 3;
      double wi = w_data;
      cout << i << " " << dstart << " " << roffset << endl;

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

    // prior term, i.e. similarity to source mesh
    cout << "prior terms ..." << endl;
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

    // Laplacian distortion term
    cout << "Laplacian terms ..." << endl;
    for (int i = 0, ioffset = 0; i < nverts; ++i) {
      auto& Ti = Tm[i];
      auto& Ni = N[i];
      double wi = w_dist;

      for (int k = ioffset; k < ioffset+3; ++k) {
        // deformation part
        M.append(roffset+k, ioffset+0, (1 - Ti(k, 0))*wi);
        M.append(roffset+k, ioffset+1, (0 - Ti(k, 1))*wi);
        M.append(roffset+k, ioffset+2, (0 - Ti(k, 2))*wi);

        int j = 0;
        double wij = -1.0 / Ni.size();
        for (auto Nij : Ni) {
          int jstart = Nij * 3;
          int joffset = j * 3;
          M.append(roffset+k, jstart+0, (wij - Ti(k, joffset))*wi);
          M.append(roffset+k, jstart+1, (0 - Ti(k, joffset+1))*wi);
          M.append(roffset+k, jstart+2, (0 - Ti(k, joffset+2))*wi);
        }
      }

      ioffset += 3;
    }

    cout << nterms << endl;

    // solve sparse linear system
    cout << "M matrix assembled..." << endl;
    // compute M' * M
    cout << "computing M'*M..." << endl;
    auto Ms = M.to_sparse();
    auto Mt = cholmod_transpose(Ms, 2, &common);
    auto MtM = cholmod_aat(Mt, NULL, 0, 1, &common);
    MtM->stype = 1;
        
    // compute M' * b
    cout << "computing M'*b..." << endl;
    cholmod_dense *bs = cholmod_allocate_dense(Ms->nrow, 1, Ms->nrow, CHOLMOD_REAL, &common);
    memcpy(bs->x, &(b[0]), sizeof(double)*Ms->nrow);
    cholmod_dense *Mtb = cholmod_allocate_dense(MtM->nrow, 1, MtM->nrow, CHOLMOD_REAL, &common);
    double alpha[2] = {1, 0}; double beta[2] = {0, 0};

    cholmod_print_sparse(Mt, "Mt", &common);
    cholmod_print_dense(bs, "bs", &common);
    cholmod_print_dense(Mtb, "Mtb", &common);
    cholmod_sdmult(Mt, 0, alpha, beta, bs, Mtb, &common);

    cout << "Solving (M'*M)\(M'*b) ..." << endl;
    // solve (M'*M)\(M'*b)
    // solution vector
    cholmod_factor *L = cholmod_analyze(MtM, &common);
    cholmod_factorize(MtM, L, &common);
    cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, Mtb, &common);
    cout << "done." << endl;

    ofstream fout("x.txt");
    for(int xidx=0;xidx=x->nrow;++xidx) fout << ((double*)x->x) [xidx] << endl;
    fout.close();

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

  cholmod_finish(&common);
  
  
  return D;
}

BasicMesh MeshDeformer::deformWithPoints(const PointCloud &P, const PointCloud &lm_points)
{
  BasicMesh D;
  return D;
}

vector<ICPCorrespondence> MeshDeformer::findClosestPoints_bruteforce(const PointCloud &P, const BasicMesh &mesh)
{
  int nfaces = mesh.faces.nrow;
  int npoints = P.points.nrow;
  vector<ICPCorrespondence> corrs(npoints);
#pragma omp parallel for
  for (int pidx = 0; pidx < npoints; ++pidx) {
    int poffset = pidx * 3;
    double px = P.points(poffset), py = P.points(poffset+1), pz = P.points(poffset+2);

#undef max
    ICPCorrespondence bestCorr;
    bestCorr.d = numeric_limits<double>::max();

    for (int i = 0, foffset=0; i < nfaces; ++i, foffset+=3) {
      int v0 = mesh.faces(foffset), v1 = mesh.faces(foffset+1), v2 = mesh.faces(foffset+2);
      // find closest point on triangle
      ICPCorrespondence corr = findClosestPoint_triangle(px, py, pz, 
                                                         mesh.verts.rowptr(v0), 
                                                         mesh.verts.rowptr(v1), 
                                                         mesh.verts.rowptr(v2));
      corr.tidx = i;
      if (corr.d < bestCorr.d) bestCorr = corr;
    }
    // compute bary-centric coordinates
    int toffset = bestCorr.tidx * 3;
    int v0idx = mesh.faces(toffset), v1idx = mesh.faces(toffset + 1), v2idx = mesh.faces(toffset + 2);

    const double* v0 = mesh.verts.rowptr(v0idx);
    const double* v1 = mesh.verts.rowptr(v1idx);
    const double* v2 = mesh.verts.rowptr(v2idx);

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

ICPCorrespondence MeshDeformer::findClosestPoint_triangle(double px, double py, double pz, const double *v0, const double *v1, const double *v2)
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
