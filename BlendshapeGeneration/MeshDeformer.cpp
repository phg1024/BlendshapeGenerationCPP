#include "MeshDeformer.h"
#include "Geometry/geometryutils.hpp"

MeshDeformer::MeshDeformer()
{
}


MeshDeformer::~MeshDeformer()
{
}

BasicMesh MeshDeformer::deformWithMesh(const BasicMesh &T, const PointCloud &lm_points)
{
  cout << "deformation with mesh ..." << endl;
  int nverts = S.verts.rows;
  int nfaces = S.faces.rows;

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
  BasicMatrix<double> delta(nverts, 3);
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
    BasicMatrix<double> V = BasicMatrix<double>::zeros(3, 7);
    V(0) = x; V(2) = z; V(3) = -y; V(4) = 1.0;
    V(7) = y; V(9) = -z; V(10) = x; V(12) = 1.0;
    V(14) = z; V(15) = y; V(16) = -x; V(20) = 1.0;

    return V;
  };

  // compute A matrix
  vector<BasicMatrix<double>> A(nverts);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    A[i] = BasicMatrix<double>::zeros(3*(Ni.size()+1), 7);
    
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
    BasicMatrix<double> V = BasicMatrix<double>::zeros(3, 7);
    V(0) = x; V(2) = z; V(3) = -y;
    V(7) = y; V(9) = -z; V(10) = x;
    V(14) = z; V(15) = y; V(16) = -x;

    return V;
  };

  // compute T matrix
  vector<BasicMatrix<double>> Tm(nverts);
  for (int i = 0; i < nverts; ++i) {
    int offset_i = i * 3;
    auto Di = makeDMatrix(delta(offset_i), delta(offset_i+1), delta(offset_i+2));
    auto& Ai = A[i];
    auto At = A[i].transposed();
    Tm[i] = Di * ((At * Ai).inv() * At);
  }

  BasicMesh D = S.clone(); // make a copy of the source mesh

  // generate a point cloud from the target mesh  
  int npoints = 0;
  for (int i = 0; i < T.verts.rows; ++i) {
    if (T.verts(i * 3 + 2) > -0.1) ++npoints;
  }
  PointCloud P;
  P.points.resize(npoints, 3);
  for (int i = 0, ridx=0; i < T.verts.rows; ++i) {
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

  double w_icp = 0.0, w_icp_step = landmarks.size() / (double)npoints;
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
    nrows += ndata;

    // add prior terms
    int nprior = S.verts.rows;
    nterms += nprior * 3;
    nrows += nprior;

    // add distortion terms
    nterms += ndistortion;
    nrows += S.verts.rows * 3;

    SparseMatrix<double> M(nrows, nverts*3);
    M.resize(nterms);
    vector<double> b(nterms, 0);
    int midx = 0;
    int roffset = 0;

    // ICP term
    for (int i = 0; i < npoints; ++i) {
      double wi = icp_corr[i].weight * w_icp;
      int toffset = icp_corr[i].tidx*3;
      int v0 = S.faces(toffset), v1 = S.faces(toffset + 1), v2 = S.faces(toffset + 2);

      int v0offset = v0 * 3, v1offset = v1 * 3, v2offset = v2 * 3;
      double wi0 = icp_corr[i].bcoords[0] * wi;
      double wi1 = icp_corr[i].bcoords[1] * wi;
      double wi2 = icp_corr[i].bcoords[2] * wi;
      
      int ioffset = i * 3;
      M.Ai[midx] = ioffset; M.Aj[midx] = v0offset + 0; M.Av[midx] = wi0; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v1offset + 0; M.Av[midx] = wi1; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v2offset + 0; M.Av[midx] = wi2; ++midx;
      b[ioffset] = P.points(ioffset) * wi;
      ++ioffset;

      M.Ai[midx] = ioffset; M.Aj[midx] = v0offset + 1; M.Av[midx] = wi0; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v1offset + 1; M.Av[midx] = wi1; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v2offset + 1; M.Av[midx] = wi2; ++midx;
      b[ioffset] = P.points(ioffset) * wi;
      ++ioffset;
      
      M.Ai[midx] = ioffset; M.Aj[midx] = v0offset + 2; M.Av[midx] = wi0; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v1offset + 2; M.Av[midx] = wi1; ++midx;
      M.Ai[midx] = ioffset; M.Aj[midx] = v2offset + 2; M.Av[midx] = wi2; ++midx;
      b[ioffset] = P.points(ioffset) * wi;
      ++ioffset;
    }
    roffset += npoints * 3;

    // landmarks term term
    for (int i = 0, ioffset = 0; i < ndata; ++i) {
      int dstart = landmarks[i] * 3;
      double wi = w_data;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 1; M.Av[midx] = wi; ++midx;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 2; M.Av[midx] = wi; ++midx;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 3; M.Av[midx] = wi; ++midx;

      b[roffset + ioffset] = lm_points.points(ioffset) * wi; ++ioffset;
      b[roffset + ioffset] = lm_points.points(ioffset) * wi; ++ioffset;
      b[roffset + ioffset] = lm_points.points(ioffset) * wi; ++ioffset;
    }
    roffset += ndata;

    // prior term, i.e. similarity to source mesh
    for (int i = 0, ioffset = 0; i < nprior; ++i) {
      int dstart = i * 3;
      double wi = w_prior;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 1; M.Av[midx] = wi; ++midx;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 2; M.Av[midx] = wi; ++midx;
      M.Ai[midx] = roffset + i; M.Aj[midx] = dstart + 3; M.Av[midx] = wi; ++midx;

      b[roffset + ioffset] = S.verts(ioffset) * wi; ++ioffset;
      b[roffset + ioffset] = S.verts(ioffset) * wi; ++ioffset;
      b[roffset + ioffset] = S.verts(ioffset) * wi; ++ioffset;
    }
    roffset += nprior;

    // Laplacian distortion term
    for (int i = 0, ioffset = 0; i < nverts; ++i) {
      auto& Ti = Tm[i];
      auto& Ni = N[i];
      double wi = w_dist;

      for (int k = 0; k < 3; ++k) {
        // deformation part
        M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = ioffset + 0; M.Av[midx] = (1 - Ti(k, 0))*wi; midx = midx + 1;
        M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = ioffset + 1; M.Av[midx] = -Ti(k, 1)*wi; midx = midx + 1;
        M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = ioffset + 2; M.Av[midx] = -Ti(k, 2)*wi; midx = midx + 1;

        int j = 0;
        double wij = -1.0 / Ni.size();
        for (auto Nij : Ni) {
          int jstart = Nij * 3;
          int joffset = j * 3;
          M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = jstart + 0; M.Av[midx] = (wij - Ti(k, joffset))*wi; midx = midx + 1;
          M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = jstart + 1; M.Av[midx] = -Ti(k, joffset + 1)*wi; midx = midx + 1;
          M.Ai[midx] = roffset + ioffset + k; M.Aj[midx] = jstart + 2; M.Av[midx] = -Ti(k, joffset + 2)*wi; midx = midx + 1;
        }
      }

      ioffset += 3;
    }

    // solve sparse linear system
    
    // compute M' * M
    SparseMatrix<double> MtM = M.selfProduct();
        
    // compute M' * b
    vector<double> Mtb = M * b;

    // solve (M'*M)\(M'*b)
    // solution vector
    vector<double> x = MtM.solve(Mtb, &common);

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
  int nfaces = mesh.faces.rows;
  int npoints = P.points.rows;
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
