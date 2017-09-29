#include "meshdeformer.h"

#include <MultilinearReconstruction/utils.hpp>
#include <MultilinearReconstruction/OffscreenMeshVisualizer.h>

#include "Geometry/geometryutils.hpp"

#include "cereswrapper.h"
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

#include "unsupported/Eigen/SparseExtra"

using Eigen::CholmodSupernodalLLT;
using Eigen::Success;

MeshDeformer::MeshDeformer() {}

MeshDeformer::~MeshDeformer() {}

BasicMesh MeshDeformer::deformWithMesh(const BasicMesh &T, const MatrixX3d &lm_points, int itmax)
{
  MatrixX3d P = T.samplePoints(8, -0.1);
  return deformWithPoints(P, lm_points, itmax);
}

BasicMesh MeshDeformer::deformWithPoints(const MatrixX3d &P, const MatrixX3d &lm_points, int itmax)
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
  MatrixXd delta(nverts, 3);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];

    Vector3d Si(0, 0, 0);
    for (auto j : Ni) {
      Si += S.vertex(j);
    }

    delta.row(i) = S.vertex(i) - Si / static_cast<double>(Ni.size());
  }

  auto makeVMatrix = [](double x, double y, double z) {
    MatrixXd V = MatrixXd::Zero(3, 7);
    /*
     *     x   0  z -y 1 0 0
     *     y  -z  0  x 0 1 0
     *     z   y -x  0 0 0 1
     */
    V(0, 0) =  x; V(1, 0) =  y;  V(2, 0) =  z;
                  V(1, 1) = -z;  V(2, 1) =  y;
    V(0, 2) =  z;                V(2, 2) = -x;
    V(0, 3) = -y; V(1, 3) = x;
    V(0, 4) = 1;
    V(1, 5) = 1;
    V(2, 6) = 1;

    return V;
  };

  vector<MatrixXd> Vs(nverts);
  for(int i = 0; i < nverts; ++i) {
    auto vert_i = S.vertex(i);
    Vs[i] = makeVMatrix(vert_i[0], vert_i[1], vert_i[2]);
  }

  // @TODO precompute all V matrices
  // compute A matrix
  vector<MatrixXd> A(nverts);
  for (int i = 0; i < nverts; ++i) {
    auto& Ni = N[i];
    A[i] = MatrixXd::Zero(3*(Ni.size()+1), 7);

    // set the vertex's terms
    A[i].topRows(3) = Vs[i];

    // set the neighbor terms
    int roffset = 3;
    for (auto j : Ni) {
      A[i].middleRows(roffset, 3) = Vs[j];
      roffset += 3;
    }
  }

  auto makeDMatrix = [](double x, double y, double z) {
    MatrixXd D = MatrixXd::Zero(3, 7);
    /*
     *     x   0  z -y 0 0 0
     *     y  -z  0  x 0 0 0
     *     z   y -x  0 0 0 0
     */
    D(0, 0) =  x; D(1, 0) =  y;  D(2, 0) =  z;
                  D(1, 1) = -z;  D(2, 1) =  y;
    D(0, 2) =  z;                D(2, 2) = -x;
    D(0, 3) = -y; D(1, 3) = x;

    return D;
  };

  // compute T matrix
  vector<MatrixXd> Tm(nverts);
  for (int i = 0; i < nverts; ++i) {
    int offset_i = i * 3;
    MatrixXd Di = makeDMatrix(delta(i, 0), delta(i, 1), delta(i, 2));
    auto& Ai = A[i];
    #if 0
    MatrixXd At = Ai.transpose();
    MatrixXd AtAi = At * Ai;
    //cout << AtAi << endl;
    MatrixXd invAtAi = AtAi.inverse();
    //cout << invAtAi << endl;
    Tm[i] = Di * (invAtAi * At);
    #else
    Tm[i] = Di * ((Ai.transpose() * Ai).inverse() * Ai.transpose());
    #endif
    //cout << Tm[i] << endl;
  }

  int npoints = P.rows();

  //ofstream fP("P.txt");
  //fP << P;
  //fP.close();

  using Tripletd = Eigen::Triplet<double>;
  using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;

  // pre-compute Laplacian part of M
#define USE_PRECOMPUTED_LAPLACIAN_TERM 1
#if USE_PRECOMPUTED_LAPLACIAN_TERM
  // Establish vertex-index mapping
  vertex_index_map.clear();
  vertex_index_map.resize(S.NumVertices(), -1);
  int vertex_index = 0;

  // Add valid vertices
  vector<int> inverse_vertex_index_map;
  for(auto vi : valid_vertices) {
    vertex_index_map[vi] = vertex_index;
    inverse_vertex_index_map.push_back(vi);
    ++vertex_index;
  }

  // Add boundary vertices
  for(auto vi : fixed_faces_boundary_vertices) {
    if( vertex_index_map[vi] == -1 ) {
      vertex_index_map[vi] = vertex_index;
      inverse_vertex_index_map.push_back(vi);
      ++vertex_index;
    }
  }

  // Add landmarks vertices
  // Assume the landmarks are vertices of the valid faces
  for(auto vi : landmarks) {
    if( vertex_index_map[vi] == -1 ) {
      vertex_index_map[vi] = vertex_index;
      inverse_vertex_index_map.push_back(vi);
      ++vertex_index;
    }
  }
  const int ncols = vertex_index * 3;

  //cout << "ncols = " << ncols << endl;

  // count the total number of terms
  int nrows = 0;
  int nterms = 0;

  // add ICP terms
  nterms += npoints * 9;
  nrows += npoints * 3;

  //cout << "npoints = " << npoints << endl;

  // add landmarks terms
  int ndata = landmarks.size();
  nterms += ndata * 3;
  nrows += ndata * 3;

  //cout << "ndata = " << ndata << endl;

  // add prior terms
  //int nprior = S.NumVertices();
  int nprior = fixed_faces_boundary_vertices.size();
  nterms += nprior * 3;
  nrows += nprior * 3;

  //cout << "nprior = " << nprior << endl;

  // add distortion terms
  // the number of matrix elements in distortion term
  int ndistortion = 0;
  for (int i=0;i<nverts;++i) {
    if(vertex_index_map[i] >= 0) {
      auto& Ni = N[i];
      ndistortion += (Ni.size() + 1);
    }
  }
  ndistortion *= 9;

  //cout << "ndistortion = " << valid_vertices.size() << endl;

  nterms += ndistortion;
  //nrows += S.NumVertices() * 3;
  nrows += valid_vertices.size() * 3;

  // add rigidity term
  int nrigidity = 0;
  for (int i=0;i<nverts;++i) {
    if(vertex_index_map[i] >= 0) {
      auto& Ni = N[i];
      nrigidity += Ni.size();
    }
  }

  nterms += nrigidity * 6;
  nrows += nrigidity * 3;

  SparseMatrixd M_lap(valid_vertices.size()*3, ncols);
  SparseMatrixd M_lap_T, M_lapTM_lap;

  // TODO: remember to mul w_dist back to M_lap_T and w_dist^2 to M_lapTM_lap
  {
    // Laplacian distortion term
    vector<Tripletd> M_coeffs;
    M_coeffs.reserve(100000);

    PhGUtils::Timer tdist;
    tdist.tic();
    int roffset = 0;
    //cout << "assembling Laplacian terms ..." << endl;
    for (int i = 0; i < valid_vertices.size(); ++i) {
      int vi = valid_vertices[i];
      auto& Ti = Tm[vi];
      auto& Ni = N[vi];
      double wi = 1.0;

      //cout << Ti << endl;

      //int istart = i * 3;
      int istart = vertex_index_map[vi] * 3;
      assert(istart >= 0);

      // deformation part
      //M.append(roffset+istart+0, istart+0, wi);
      //M.append(roffset+istart+1, istart+1, wi);
      //M.append(roffset+istart+2, istart+2, wi);

      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+0, (1 - Ti(0, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+1, (0 - Ti(0, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+2, (0 - Ti(0, 2))*wi));

      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+0, (0 - Ti(1, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+1, (1 - Ti(1, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+2, (0 - Ti(1, 2))*wi));

      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+0, (0 - Ti(2, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+1, (0 - Ti(2, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+2, (1 - Ti(2, 2))*wi));

      int j = 1;
      double wij = -1.0 / Ni.size();
      for (auto Nij : Ni) {
        //int jstart = Nij * 3;
        int jstart = vertex_index_map[Nij] * 3;
        assert(jstart >= 0);
        int joffset = j * 3; ++j;
        //M.append(roffset+istart+0, jstart+0, wij*wi);
        //M.append(roffset+istart+1, jstart+1, wij*wi);
        //M.append(roffset+istart+2, jstart+2, wij*wi);

        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+0, (wij - Ti(0, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+1, (  0 - Ti(0, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+2, (  0 - Ti(0, joffset+2))*wi));

        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+0, (  0 - Ti(1, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+1, (wij - Ti(1, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+2, (  0 - Ti(1, joffset+2))*wi));

        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+0, (  0 - Ti(2, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+1, (  0 - Ti(2, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+2, (wij - Ti(2, joffset+2))*wi));
      }
    }

    M_lap.setFromTriplets(M_coeffs.begin(), M_coeffs.end());
    M_lap_T = M_lap.transpose();
    M_lapTM_lap = (M_lap_T * M_lap).pruned();

    tdist.toc("assembling distortion term");
  }
  #endif
  cout << "check point 1" << endl;

  BasicMesh D = S; // make a copy of the source mesh

  // main deformation loop
  int iters = 0;

  double ratio_data2icp = max(0.1, 10.0*lm_points.rows() / (double) S.NumVertices());
  //cout << ratio_data2icp << endl;
  double w_icp = 0, w_icp_step = ratio_data2icp;
  double w_data = 10.0*74.0/lm_points.rows(), w_data_step = w_data/itmax;
  double w_dist = 10000.0 * ratio_data2icp, w_dist_step = w_dist/itmax;
  double w_prior = 10.0, w_prior_step = w_prior*0.95/itmax;

  #define ANALYZE_ONCE 1  // analyze MtM only once
  bool analyzed = false;
  //CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
  CholmodSupernodalLLT<SparseMatrixd> solver;

  while (iters++ < itmax) {
    cout << "iteration " << iters << endl;

    // find correspondence
    PhGUtils::Timer tcorr;
    tcorr.tic();
    //vector<ICPCorrespondence> icp_corr = findClosestPoints_bruteforce(P, D);
    //vector<ICPCorrespondence> icp_corr = findClosestPoints_projection(P, D);
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

#if USE_PRECOMPUTED_LAPLACIAN_TERM
    // Nothing to do here
#else
    // Establish vertex-index mapping
    vertex_index_map.clear();
    vertex_index_map.resize(S.NumVertices(), -1);
    int vertex_index = 0;

    // Add valid vertices
    vector<int> inverse_vertex_index_map;
    for(auto vi : valid_vertices) {
      vertex_index_map[vi] = vertex_index;
      inverse_vertex_index_map.push_back(vi);
      ++vertex_index;
    }

    // Add boundary vertices
    for(auto vi : fixed_faces_boundary_vertices) {
      if( vertex_index_map[vi] == -1 ) {
        vertex_index_map[vi] = vertex_index;
        inverse_vertex_index_map.push_back(vi);
        ++vertex_index;
      }
    }

    // Add landmarks vertices
    // Assume the landmarks are vertices of the valid faces
    for(auto vi : landmarks) {
      if( vertex_index_map[vi] == -1 ) {
        vertex_index_map[vi] = vertex_index;
        inverse_vertex_index_map.push_back(vi);
        ++vertex_index;
      }
    }
    const int ncols = vertex_index * 3;

    //cout << "ncols = " << ncols << endl;

    // count the total number of terms
    int nrows = 0;
    int nterms = 0;

    // add ICP terms
    nterms += npoints * 9;
    nrows += npoints * 3;

    //cout << "npoints = " << npoints << endl;

    // add landmarks terms
    int ndata = landmarks.size();
    nterms += ndata * 3;
    nrows += ndata * 3;

    //cout << "ndata = " << ndata << endl;

    // add prior terms
    //int nprior = S.NumVertices();
    int nprior = fixed_faces_boundary_vertices.size();
    nterms += nprior * 3;
    nrows += nprior * 3;

    //cout << "nprior = " << nprior << endl;

    // add distortion terms
    // the number of matrix elements in distortion term
    int ndistortion = 0;
    for (int i=0;i<nverts;++i) {
      if(vertex_index_map[i] >= 0) {
        auto& Ni = N[i];
        ndistortion += (Ni.size() + 1);
      }
    }
    ndistortion *= 9;

    //cout << "ndistortion = " << valid_vertices.size() << endl;

    nterms += ndistortion;
    //nrows += S.NumVertices() * 3;
    nrows += valid_vertices.size() * 3;
#endif

    //cout << "nterms = " << nterms << endl;
    //cout << "nrows = " << nrows << endl;

    vector<Tripletd> M_coeffs;
    M_coeffs.reserve(nterms);

    SparseMatrixd M(nrows, ncols);
    VectorXd b = VectorXd::Zero(nrows);
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

      //int v0offset = v0 * 3, v1offset = v1 * 3, v2offset = v2 * 3;
      int v0offset = vertex_index_map[v0] * 3; assert(v0offset>=0);
      int v1offset = vertex_index_map[v1] * 3; assert(v1offset>=0);
      int v2offset = vertex_index_map[v2] * 3; assert(v2offset>=0);

      double wi0 = icp_corr[i].bcoords[0] * wi;
      double wi1 = icp_corr[i].bcoords[1] * wi;
      double wi2 = icp_corr[i].bcoords[2] * wi;

      M_coeffs.push_back(Tripletd(roffset, v0offset, wi0));
      M_coeffs.push_back(Tripletd(roffset, v1offset, wi1));
      M_coeffs.push_back(Tripletd(roffset, v2offset, wi2));
      b[roffset] = P(i, 0) * wi;
      ++roffset;

      M_coeffs.push_back(Tripletd(roffset, v0offset+1, wi0));
      M_coeffs.push_back(Tripletd(roffset, v1offset+1, wi1));
      M_coeffs.push_back(Tripletd(roffset, v2offset+1, wi2));
      b[roffset] = P(i, 1) * wi;
      ++roffset;

      M_coeffs.push_back(Tripletd(roffset, v0offset+2, wi0));
      M_coeffs.push_back(Tripletd(roffset, v1offset+2, wi1));
      M_coeffs.push_back(Tripletd(roffset, v2offset+2, wi2));
      b[roffset] = P(i, 2) * wi;
      ++roffset;
    }
    ticp.toc("assembling ICP term");

    #define USE_LANDMARKS 1
    #if USE_LANDMARKS
    {
      // landmarks term term
      //cout << "assembling landmarks terms ..." << endl;
      PhGUtils::Timer tland;
      tland.tic();
      for (int i = 0, ioffset=0; i < ndata; ++i) {
        //int dstart = landmarks[i] * 3;
        int dstart = vertex_index_map[landmarks[i]] * 3;
        assert(dstart >= 0);

        double wi = w_data;

        /*
        // for debugging
        cout << "("
             << lm_points(i, 0) << ","
             << lm_points(i, 1) << ","
             << lm_points(i, 2)
             << ")"
             << " vs "
             << "("
             << S.vertex(landmarks[i])[0] << ","
             << S.vertex(landmarks[i])[1] << ","
             << S.vertex(landmarks[i])[2]
             << ") " << wi
             << endl;
        */

        M_coeffs.push_back(Tripletd(roffset, dstart, wi)); ++dstart;
        b[roffset] = lm_points(i, 0) * wi;
        ++roffset; ++ioffset;

        M_coeffs.push_back(Tripletd(roffset, dstart, wi)); ++dstart;
        b[roffset] = lm_points(i, 1) * wi;
        ++roffset; ++ioffset;

        const double weight_z = 0;
        M_coeffs.push_back(Tripletd(roffset, dstart, wi * weight_z));
        b[roffset] = lm_points(i, 2) * wi * weight_z;
        ++roffset; ++ioffset;
      }
      tland.toc("assembling landmarks term");
    }
    #endif

    // prior term, i.e. similarity to source mesh
    //cout << "assembling prior terms ..." << endl;
    PhGUtils::Timer tprior;
    tprior.tic();
    #if 0
    for (int i = 0, ioffset = 0; i < nprior; ++i) {
      double wi = w_prior;
      auto vert_i = S.vertex(i);
      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[0] * wi;
      ++roffset; ++ioffset;

      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[1] * wi;
      ++roffset; ++ioffset;

      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[2] * wi;
      ++roffset; ++ioffset;
    }
    #else
    for (int i = 0; i < nprior; ++i) {
      int vi = fixed_faces_boundary_vertices[i];
      double wi = w_prior;
      auto vert_i = S.vertex(vi);
      int ioffset = vertex_index_map[vi] * 3;
      assert(ioffset >= 0);

      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[0] * wi;
      ++roffset; ++ioffset;

      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[1] * wi;
      ++roffset; ++ioffset;

      M_coeffs.push_back(Tripletd(roffset, ioffset, wi));
      b[roffset] = vert_i[2] * wi;
      ++roffset; ++ioffset;
    }
    #endif
    tprior.toc("assembling prior term");

    // Rigidity term
    {
      PhGUtils::Timer trigid;
      trigid.tic();

      const double w_rigid = 1.0;
      for (int i = 0; i < valid_vertices.size(); ++i) {
        int vi = valid_vertices[i];
        auto& Ti = Tm[vi];
        auto& Ni = N[vi];
        double wi = w_dist;

        int istart = vertex_index_map[vi] * 3;
        assert(istart >= 0);

        const auto& vertex_vi = S.vertex(vi);

        int j = 1;
        double wij = -1.0 / Ni.size();
        for (auto Nij : Ni) {
          // FIXME use cotagent weights
          const double wij = 1.0;
          int jstart = vertex_index_map[Nij] * 3;
          assert(jstart >= 0);

          const auto& vertex_vj = S.vertex(Nij);

          M_coeffs.push_back(Tripletd(roffset+0, istart+0, wij*w_rigid));
          M_coeffs.push_back(Tripletd(roffset+1, istart+1, wij*w_rigid));
          M_coeffs.push_back(Tripletd(roffset+2, istart+2, wij*w_rigid));

          M_coeffs.push_back(Tripletd(roffset+0, jstart+0, -wij*w_rigid));
          M_coeffs.push_back(Tripletd(roffset+1, jstart+1, -wij*w_rigid));
          M_coeffs.push_back(Tripletd(roffset+2, jstart+2, -wij*w_rigid));

          auto vi_m_vj = vertex_vi - vertex_vj;
          b[roffset+0] = vi_m_vj[0];
          b[roffset+1] = vi_m_vj[1];
          b[roffset+2] = vi_m_vj[2];

          roffset += 3;
        }
      }

      trigid.toc("assembling rigidity term");
    }

#if USE_PRECOMPUTED_LAPLACIAN_TERM
    // Nothing to do here since it's pre-computed
#else
    // Laplacian distortion term
    PhGUtils::Timer tdist;
    tdist.tic();
    //cout << "assembling Laplacian terms ..." << endl;
    for (int i = 0; i < valid_vertices.size(); ++i) {
      int vi = valid_vertices[i];
      auto& Ti = Tm[vi];
      auto& Ni = N[vi];
      double wi = w_dist;

      //cout << Ti << endl;

      //int istart = i * 3;
      int istart = vertex_index_map[vi] * 3;
      assert(istart >= 0);

      // deformation part
      //M.append(roffset+istart+0, istart+0, wi);
      //M.append(roffset+istart+1, istart+1, wi);
      //M.append(roffset+istart+2, istart+2, wi);

      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+0, (1 - Ti(0, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+1, (0 - Ti(0, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+0, istart+2, (0 - Ti(0, 2))*wi));

      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+0, (0 - Ti(1, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+1, (1 - Ti(1, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+1, istart+2, (0 - Ti(1, 2))*wi));

      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+0, (0 - Ti(2, 0))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+1, (0 - Ti(2, 1))*wi));
      M_coeffs.push_back(Tripletd(roffset+istart+2, istart+2, (1 - Ti(2, 2))*wi));

      int j = 1;
      double wij = -1.0 / Ni.size();
      for (auto Nij : Ni) {
        //int jstart = Nij * 3;
        int jstart = vertex_index_map[Nij] * 3;
        assert(jstart >= 0);
        int joffset = j * 3; ++j;
        //M.append(roffset+istart+0, jstart+0, wij*wi);
        //M.append(roffset+istart+1, jstart+1, wij*wi);
        //M.append(roffset+istart+2, jstart+2, wij*wi);

        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+0, (wij - Ti(0, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+1, (  0 - Ti(0, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+0, jstart+2, (  0 - Ti(0, joffset+2))*wi));

        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+0, (  0 - Ti(1, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+1, (wij - Ti(1, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+1, jstart+2, (  0 - Ti(1, joffset+2))*wi));

        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+0, (  0 - Ti(2, joffset+0))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+1, (  0 - Ti(2, joffset+1))*wi));
        M_coeffs.push_back(Tripletd(roffset+istart+2, jstart+2, (wij - Ti(2, joffset+2))*wi));
      }
    }
    tdist.toc("assembling distortion term");
#endif

    #if 0
    {
      ofstream fout("M.txt");
      for(int i=0;i<M_coeffs.size();++i) {
        fout << M_coeffs[i].row() << " " << M_coeffs[i].col() << " " << M_coeffs[i].value() << endl;
      }
      fout.close();
    }
    #endif

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
    M.setFromTriplets(M_coeffs.begin(), M_coeffs.end());

    SparseMatrixd Mt = M.transpose();
    //SparseMatrixd M_lapTM_lap_weighted = M_lapTM_lap * (w_dist * w_dist);
    SparseMatrixd MtM = (Mt * M).pruned();
    MtM += M_lapTM_lap * (w_dist * w_dist);

    // compute M' * b
    cout << "computing M'*b..." << endl;
    VectorXd Mtb = Mt * b;
    tmatrix.toc("constructing linear equations");

    PhGUtils::Timer tsolve;
    tsolve.tic();

    VectorXd x;

    // FIXME try QR factorization
    bool use_direct_solver = true;
    if(use_direct_solver) {
      cout << "Using direct solver..." << endl;
      cout << "Solving (M'*M)\\(M'*b) ..." << endl;
      cout << MtM.rows() << 'x' << MtM.cols() << endl;
      cout << MtM.nonZeros() << endl;

/*
      {
        Eigen::saveMarket(MtM, "MtM.mtx");
        cout << "done" << endl;
        exit(0);
      }
*/
      // solve (M'*M)\(M'*b)
      // solution vector
      #if ANALYZE_ONCE
        if(!analyzed) {
          solver.analyzePattern(MtM);
          analyzed = true;
        }
        solver.factorize(MtM);
      #else
        solver.compute(MtM);
      #endif

      if(solver.info()!=Eigen::Success) {
        cerr << "Failed to decompose matrix A." << endl;
        exit(-1);
      }

      x = solver.solve(Mtb);
      if(solver.info()!=Success) {
        cerr << "Failed to solve A\\b." << endl;
        exit(-1);
      }
    }

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
    for(int i=0;i<nverts;++i) {
      D.set_vertex(i, Vector3d(x[i*3+0],
                               x[i*3+1],
                               x[i*3+2]));
    }
    #else
    for(int i=0;i<inverse_vertex_index_map.size();++i) {
      D.set_vertex(inverse_vertex_index_map[i], Vector3d(x[i*3+0],
                                                         x[i*3+1],
                                                         x[i*3+2]));
    }
    #endif

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
}

namespace {
struct NormalCostFunction {
  NormalCostFunction(const BasicMesh& mesh,
                     const vector<set<int>>& incident_faces,
                     const map<int, int>& ptrs_mapper,
                     const NormalConstraint& constraint,
                     double weight)
    : mesh(mesh), incident_faces(incident_faces), ptrs_mapper(ptrs_mapper),
      constraint(constraint), weight(weight) {}

  Vector3d computeNormal(const double *const *params, int vidx) const {
    auto incident_faces_i = incident_faces[vidx];

    Vector3d n(0, 0, 0);
    double area_sum = 0;
    // Compute the face normal for each of these faces
    for(auto i : incident_faces_i) {
      auto face_i = mesh.face(i);

      const double* params_0 = params[ptrs_mapper.at(face_i[0])];
      Vector3d v0(*params_0, *(params_0+1), *(params_0+2));

      const double* params_1 = params[ptrs_mapper.at(face_i[1])];
      Vector3d v1(*params_1, *(params_1+1), *(params_1+2));

      const double* params_2 = params[ptrs_mapper.at(face_i[2])];
      Vector3d v2(*params_2, *(params_2+1), *(params_2+2));

      auto v0v1 = v1 - v0;
      auto v0v2 = v2 - v0;
      auto n_i = v0v1.cross(v0v2);
      double area = n.norm();

      n += n_i;
      area_sum += area;
    }

    return n / area_sum;
  }

  bool operator()(const double *const *params, double *residual) const {
    // Compute the current normal at this point
    auto face_i = mesh.face(constraint.fidx);
    Vector3d n0 = computeNormal(params, face_i[0]);
    Vector3d n1 = computeNormal(params, face_i[1]);
    Vector3d n2 = computeNormal(params, face_i[2]);

    Vector3d normal_i = n0 * constraint.bcoords[0]
                      + n1 * constraint.bcoords[1]
                      + n2 * constraint.bcoords[2];

    auto print_vector = [](Vector3d v) {
      cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")" << endl;
    };

    residual[0] = (normal_i.normalized() - constraint.n).norm() * weight;

    /*
    print_vector(normal_i);
    print_vector(constraint.n);
    cout << weight << ", " << residual[0] << endl;
     */

    return true;
  }

  const BasicMesh& mesh;
  const vector<set<int>>& incident_faces;
  map<int, int> ptrs_mapper;
  NormalConstraint constraint;
  double weight;
};

struct PointCostFunction {
  PointCostFunction(Vector3d p0, double weight) : p0(p0), weight(weight) {}

  template <typename T>
  bool operator()(const T* const params, T* residual) const {
    residual[0] = (T(p0[0]) - params[0]) * T(weight);
    residual[1] = (T(p0[1]) - params[1]) * T(weight);
    residual[2] = (T(p0[2]) - params[2]) * T(weight);

    return true;
  }

  Vector3d p0;
  double weight;
};

struct DistortionCostFunction {
  DistortionCostFunction(const MatrixXd& Tm, const set<int>& neighbors, double weight)
    : Tm(Tm), neighbors(neighbors), weight(weight) {}

  bool operator()(const double *const *params, double *residual) const {
    Matrix3d I = Matrix3d::Identity();
    Vector3d ivec = (I - Tm.block(0, 0, 3, 3)) * Vector3d(params[0][0], params[0][1], params[0][2]);

    double wij = -1.0 / neighbors.size();
    int jidx = 1;
    Vector3d jsum(0, 0, 0);
    for(auto j : neighbors) {
      Vector3d jvec = (I * wij - Tm.block(0, jidx*3, 3, 3))
                    * Vector3d(params[jidx][0],params[jidx][1],params[jidx][2]);
      jsum += jvec;
      ++jidx;
    }

    residual[0] = ivec[0] + jsum[0];
    residual[1] = ivec[1] + jsum[1];
    residual[2] = ivec[2] + jsum[2];

    return true;
  }

  mutable MatrixXd Tm;
  set<int> neighbors;
  double weight;
};


  struct GraphCostFunction {
    GraphCostFunction(Vector3d ref, double weight) : ref(ref), weight(weight) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, T* residual) const {
      residual[0] = (p0[0] - p1[0] - T(ref[0])) * T(weight);
      residual[1] = (p0[1] - p1[1] - T(ref[1])) * T(weight);
      residual[2] = (p0[2] - p1[2] - T(ref[2])) * T(weight);

      return true;
    }

    Vector3d ref;
    double weight;
  };
}

BasicMesh MeshDeformer::deformWithNormals(
  const vector<NormalConstraint>& normals,
  const MatrixX3d& lm_points,
  int itmax) {
    //cout << "deformation with mesh ..." << endl;
    int nverts = S.NumVertices();
    int nfaces = S.NumFaces();

    // find the neighbor information of every vertex in the source mesh
    vector<set<int>> neighbors(nverts);
    vector<set<int>> incident_faces(nverts);
    for (int i = 0; i < nfaces; ++i) {
      auto Fi = S.face(i);
      int v1 = Fi[0], v2 = Fi[1], v3 = Fi[2];

      neighbors[v1].insert(v2); neighbors[v1].insert(v3);
      neighbors[v2].insert(v1); neighbors[v2].insert(v3);
      neighbors[v3].insert(v1); neighbors[v3].insert(v2);

      incident_faces[v1].insert(i);
      incident_faces[v2].insert(i);
      incident_faces[v3].insert(i);
    }

    // compute delta_i
    MatrixXd delta(nverts, 3);
    for (int i = 0; i < nverts; ++i) {
      auto& Ni = neighbors[i];

      Vector3d Si(0, 0, 0);
      for (auto j : Ni) {
        Si += S.vertex(j);
      }

      delta.row(i) = S.vertex(i) - Si / static_cast<double>(Ni.size());
    }

    auto makeVMatrix = [](double x, double y, double z) {
      MatrixXd V = MatrixXd::Zero(3, 7);
      /*
       *     x   0  z -y 1 0 0
       *     y  -z  0  x 0 1 0
       *     z   y -x  0 0 0 1
       */
      V(0, 0) =  x; V(1, 0) =  y;  V(2, 0) =  z;
                    V(1, 1) = -z;  V(2, 1) =  y;
      V(0, 2) =  z;                V(2, 2) = -x;
      V(0, 3) = -y; V(1, 3) = x;
      V(0, 4) = 1;
      V(1, 5) = 1;
      V(2, 6) = 1;

      return V;
    };

    vector<MatrixXd> Vs(nverts);
    for(int i = 0; i < nverts; ++i) {
      auto vert_i = S.vertex(i);
      Vs[i] = makeVMatrix(vert_i[0], vert_i[1], vert_i[2]);
    }

    // @TODO precompute all V matrices
    // compute A matrix
    vector<MatrixXd> A(nverts);
    for (int i = 0; i < nverts; ++i) {
      auto& Ni = neighbors[i];
      A[i] = MatrixXd::Zero(3*(Ni.size()+1), 7);

      // set the vertex's terms
      A[i].topRows(3) = Vs[i];

      // set the neighbor terms
      int roffset = 3;
      for (auto j : Ni) {
        A[i].middleRows(roffset, 3) = Vs[j];
        roffset += 3;
      }
    }

    auto makeDMatrix = [](double x, double y, double z) {
      MatrixXd D = MatrixXd::Zero(3, 7);
      /*
       *     x   0  z -y 0 0 0
       *     y  -z  0  x 0 0 0
       *     z   y -x  0 0 0 0
       */
      D(0, 0) =  x; D(1, 0) =  y;  D(2, 0) =  z;
                    D(1, 1) = -z;  D(2, 1) =  y;
      D(0, 2) =  z;                D(2, 2) = -x;
      D(0, 3) = -y; D(1, 3) = x;

      return D;
    };

    // compute T matrix
    vector<MatrixXd> Tm(nverts);
    for (int i = 0; i < nverts; ++i) {
      int offset_i = i * 3;
      MatrixXd Di = makeDMatrix(delta(i, 0), delta(i, 1), delta(i, 2));
      auto& Ai = A[i];
      #if 0
      MatrixXd At = Ai.transpose();
      MatrixXd AtAi = At * Ai;
      //cout << AtAi << endl;
      MatrixXd invAtAi = AtAi.inverse();
      //cout << invAtAi << endl;
      Tm[i] = Di * (invAtAi * At);
      #else
      Tm[i] = Di * ((Ai.transpose() * Ai).inverse() * Ai.transpose());
      #endif
      //cout << Tm[i] << endl;
    }

    int npoints = normals.size();

    using Tripletd = Eigen::Triplet<double>;
    using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    // pre-compute Laplacian part of M
  #define USE_PRECOMPUTED_LAPLACIAN_TERM 1
  #if USE_PRECOMPUTED_LAPLACIAN_TERM
    // Establish vertex-index mapping
    vertex_index_map.clear();
    vertex_index_map.resize(S.NumVertices(), -1);
    int vertex_index = 0;

    // Add valid vertices
    vector<int> inverse_vertex_index_map;
    for(auto vi : valid_vertices) {
      vertex_index_map[vi] = vertex_index;
      inverse_vertex_index_map.push_back(vi);
      ++vertex_index;
    }

    // Add boundary vertices
    for(auto vi : fixed_faces_boundary_vertices) {
      if( vertex_index_map[vi] == -1 ) {
        vertex_index_map[vi] = vertex_index;
        inverse_vertex_index_map.push_back(vi);
        ++vertex_index;
      }
    }

    // Add landmarks vertices
    // Assume the landmarks are vertices of the valid faces
    for(auto vi : landmarks) {
      if( vertex_index_map[vi] == -1 ) {
        vertex_index_map[vi] = vertex_index;
        inverse_vertex_index_map.push_back(vi);
        ++vertex_index;
      }
    }
    const int ncols = vertex_index * 3;

    //cout << "ncols = " << ncols << endl;

    // count the total number of terms
    int nrows = 0;
    int nterms = 0;

    // add ICP terms
    nterms += npoints * 9;
    nrows += npoints * 3;

    //cout << "npoints = " << npoints << endl;

    // add landmarks terms
    int ndata = landmarks.size();
    nterms += ndata * 3;
    nrows += ndata * 3;

    //cout << "ndata = " << ndata << endl;

    // add prior terms
    //int nprior = S.NumVertices();
    int nprior = fixed_faces_boundary_vertices.size();
    nterms += nprior * 3;
    nrows += nprior * 3;

    //cout << "nprior = " << nprior << endl;

    // add distortion terms
    // the number of matrix elements in distortion term
    int ndistortion = 0;
    for (int i=0;i<nverts;++i) {
      if(vertex_index_map[i] >= 0) {
        auto& Ni = neighbors[i];
        ndistortion += (Ni.size() + 1);
      }
    }
    ndistortion *= 9;

    //cout << "ndistortion = " << valid_vertices.size() << endl;

    nterms += ndistortion;
    //nrows += S.NumVertices() * 3;
    nrows += valid_vertices.size() * 3;

    #endif

    BasicMesh D = S; // make a copy of the source mesh
    D.ComputeNormals();

    // main deformation loop
    int iters = 0;

    double ratio_data2icp = max(0.1, 10.0*lm_points.rows() / (double) S.NumVertices());
    //cout << ratio_data2icp << endl;
    double w_icp = 0, w_icp_step = ratio_data2icp;
    double w_data = 10.0*74.0/lm_points.rows(), w_data_step = w_data/itmax;
    double w_dist = 10000.0 * ratio_data2icp, w_dist_step = w_dist/itmax;
    double w_prior = 1.0, w_prior_step = w_prior*0.95/itmax;

    auto print_vector = [](Vector3d v) {
      cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")" << endl;
    };


    auto compute_normal = [&](const double *params, int vidx) -> Vector3d {
      auto incident_faces_i = incident_faces[vidx];

      Vector3d n(0, 0, 0);
      double area_sum = 0;
      // Compute the face normal for each of these faces
      for(auto i : incident_faces_i) {
        auto face_i = D.face(i);

        const double* params_0 = params + face_i[0] * 3;
        Vector3d v0(*params_0, *(params_0+1), *(params_0+2));

        const double* params_1 = params + face_i[1] * 3;
        Vector3d v1(*params_1, *(params_1+1), *(params_1+2));

        const double* params_2 = params + face_i[2] * 3;
        Vector3d v2(*params_2, *(params_2+1), *(params_2+2));

        auto v0v1 = v1 - v0;
        auto v0v2 = v2 - v0;
        auto n_i = v0v1.cross(v0v2);
        double area = n.norm();

        n += n_i;
        area_sum += area;
      }

      return n / area_sum;
    };

    while (iters++ < itmax) {
      cout << "iteration " << iters << endl;

      vector<double> params(D.NumVertices()*3);
      for(int i=0,offset=0;i<D.NumVertices();++i) {
        auto v_i = D.vertex(i);
        params[offset] = v_i[0];++offset;
        params[offset] = v_i[1];++offset;
        params[offset] = v_i[2];++offset;
      }

      // compute fitting error
      double Efit = 0;
      {
        boost::timer::auto_cpu_timer timer_compute_error(
          "Fitting error computation time = %w seconds.\n");
        for (int i = 0, idx = 0; i < npoints; ++i) {
          auto f_i = D.face(normals[i].fidx);

          auto vn0 = compute_normal(params.data(), f_i[0]);
          auto vn1 = compute_normal(params.data(), f_i[1]);
          auto vn2 = compute_normal(params.data(), f_i[2]);

          auto vn_i = vn0 * normals[i].bcoords[0]
                      + vn1 * normals[i].bcoords[1]
                      + vn2 * normals[i].bcoords[2];

          double Ei = (normals[i].n - vn_i.normalized()).norm();
          Efit += Ei;
          //Efit = max(Efit, Ei);
        }
      }
      Efit /= npoints;
      ColorStream(ColorOutput::Red)<< "Efit = " << Efit;

  #if USE_PRECOMPUTED_LAPLACIAN_TERM
      // Nothing to do here
  #else
      // Establish vertex-index mapping
      vertex_index_map.clear();
      vertex_index_map.resize(S.NumVertices(), -1);
      int vertex_index = 0;

      // Add valid vertices
      vector<int> inverse_vertex_index_map;
      for(auto vi : valid_vertices) {
        vertex_index_map[vi] = vertex_index;
        inverse_vertex_index_map.push_back(vi);
        ++vertex_index;
      }

      // Add boundary vertices
      for(auto vi : fixed_faces_boundary_vertices) {
        if( vertex_index_map[vi] == -1 ) {
          vertex_index_map[vi] = vertex_index;
          inverse_vertex_index_map.push_back(vi);
          ++vertex_index;
        }
      }

      // Add landmarks vertices
      // Assume the landmarks are vertices of the valid faces
      for(auto vi : landmarks) {
        if( vertex_index_map[vi] == -1 ) {
          vertex_index_map[vi] = vertex_index;
          inverse_vertex_index_map.push_back(vi);
          ++vertex_index;
        }
      }
      const int ncols = vertex_index * 3;

      //cout << "ncols = " << ncols << endl;

      // count the total number of terms
      int nrows = 0;
      int nterms = 0;

      // add ICP terms
      nterms += npoints * 9;
      nrows += npoints * 3;

      //cout << "npoints = " << npoints << endl;

      // add landmarks terms
      int ndata = landmarks.size();
      nterms += ndata * 3;
      nrows += ndata * 3;

      //cout << "ndata = " << ndata << endl;

      // add prior terms
      //int nprior = S.NumVertices();
      int nprior = fixed_faces_boundary_vertices.size();
      nterms += nprior * 3;
      nrows += nprior * 3;

      //cout << "nprior = " << nprior << endl;

      // add distortion terms
      // the number of matrix elements in distortion term
      int ndistortion = 0;
      for (int i=0;i<nverts;++i) {
        if(vertex_index_map[i] >= 0) {
          auto& Ni = neighbors[i];
          ndistortion += (Ni.size() + 1);
        }
      }
      ndistortion *= 9;

      //cout << "ndistortion = " << valid_vertices.size() << endl;

      nterms += ndistortion;
      //nrows += S.NumVertices() * 3;
      nrows += valid_vertices.size() * 3;
  #endif

      Problem problem;

      bool enable_normals_term = true;
      bool enable_landmarks_term = true;
      bool enable_prior_term = true;
      bool enable_laplacian_term = true;
      bool enable_graph_term = true;

      // normals term
      if(enable_normals_term){
        boost::timer::auto_cpu_timer time_graph(
          "Time for assembling normals term = %w seconds.\n"
        );

        for(int i=0;i<normals.size();++i) {
          auto face_i = D.face(normals[i].fidx);

          map<int, double*> ptrs;
          for(int vi=0;vi<3;++vi) {
            auto& neighbors_vi = neighbors[face_i[vi]];
            for(auto vj : neighbors_vi) {
              if(!ptrs.count(vj)) {
                ptrs.insert(make_pair(vj, params.data() + vj * 3));
              }
            }
          }

          vector<double*> params_i;
          map<int, int> ptrs_mapper;
          int loc = 0;
          for(auto p : ptrs) {
            params_i.push_back(p.second);
            ptrs_mapper.insert(make_pair(p.first, loc++));
          }

          ceres::DynamicNumericDiffCostFunction<NormalCostFunction> *func_i =
            new ceres::DynamicNumericDiffCostFunction<NormalCostFunction>(
              new NormalCostFunction(D, incident_faces, ptrs_mapper, normals[i], w_icp));

          for(int i=0;i<params_i.size();++i) func_i->AddParameterBlock(3);
          func_i->SetNumResiduals(1);
          problem.AddResidualBlock(func_i, NULL, params_i);
        }
      }

      if(enable_landmarks_term){
        // landmarks term term
        boost::timer::auto_cpu_timer time_graph(
          "Time for assembling landmarks term = %w seconds.\n"
        );
        for(int i=0;i<ndata;++i) {
          int vidx = landmarks[i];
          ceres::CostFunction* func_i = new ceres::AutoDiffCostFunction<PointCostFunction, 3, 3>(
            new PointCostFunction(S.vertex(vidx), w_data));
          problem.AddResidualBlock(func_i, NULL, params.data() + vidx*3);
        }
      }

      // prior term, i.e. similarity to source mesh
      //cout << "assembling prior terms ..." << endl;
      if(enable_prior_term){
        boost::timer::auto_cpu_timer time_graph(
          "Time for assembling prior term = %w seconds.\n"
        );

        for (int i = 0; i < nprior; ++i) {
          int vi = fixed_faces_boundary_vertices[i];
          double wi = w_prior;
          ceres::CostFunction* func_i = new ceres::AutoDiffCostFunction<PointCostFunction, 3, 3>(
            new PointCostFunction(S.vertex(vi), w_prior));
          problem.AddResidualBlock(func_i, NULL, params.data() + vi * 3);
        }
      }

      // Laplacian distortion term
      if(enable_laplacian_term){
        boost::timer::auto_cpu_timer time_graph(
          "Time for assembling Laplacian distortion term = %w seconds.\n"
        );

        for(int i=0;i<D.NumVertices();++i) {
          ceres::DynamicNumericDiffCostFunction<DistortionCostFunction> *func_i =
            new ceres::DynamicNumericDiffCostFunction<DistortionCostFunction>(
              new DistortionCostFunction(Tm[i], neighbors[i], w_dist));
          vector<double*> params_i{params.data() + i * 3};
          func_i->AddParameterBlock(3);
          for(auto j : neighbors[i]) {
            func_i->AddParameterBlock(3);
            params_i.push_back(params.data() + j * 3);
          }
          func_i->SetNumResiduals(3);
          problem.AddResidualBlock(func_i, NULL, params_i);
        }

      }

      // Graph constraints
      if(enable_graph_term){
        boost::timer::auto_cpu_timer time_graph(
          "Time for assembling graph term = %w seconds.\n"
        );

        const double w_graph = 10.0;

        for(int i=0;i<D.NumVertices();++i) {
          for(auto j : neighbors[i]) {
            ceres::CostFunction* func_ij = new ceres::AutoDiffCostFunction<GraphCostFunction, 3, 3, 3>(
              new GraphCostFunction(S.vertex(i) - S.vertex(j), w_graph)
            );
            problem.AddResidualBlock(func_ij, NULL, params.data() + i * 3, params.data() + j * 3);
          }
        }
      }

      // Sovle the problem
      {
        boost::timer::auto_cpu_timer timer_solve(
          "Problem solve time = %w seconds.\n");
        cout << "Sovling the problem ..." << endl;
        ceres::Solver::Options options;
        options.max_num_iterations = 10;
        //options.minimizer_type = ceres::LINE_SEARCH;
        options.line_search_direction_type = ceres::LBFGS;

        //options.initial_trust_region_radius = 0.25;
        //options.min_trust_region_radius = 0.001;
        //options.max_trust_region_radius = 0.5;
        //options.min_lm_diagonal = 1.0;
        //options.max_lm_diagonal = 1.0;

        options.num_threads = 4;
        options.num_linear_solver_threads = 4;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;
      }

      const auto& x = params;

      // update the vertices of D using the x vector
      #if 1
      for(int i=0;i<nverts;++i) {
        D.set_vertex(i, Vector3d(x[i*3+0],
                                 x[i*3+1],
                                 x[i*3+2]));
      }
      #else
      for(int i=0;i<inverse_vertex_index_map.size();++i) {
        D.set_vertex(inverse_vertex_index_map[i], Vector3d(x[i*3+0],
                                                           x[i*3+1],
                                                           x[i*3+2]));
      }
      #endif

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
}

vector<ICPCorrespondence> MeshDeformer::findClosestPoints_projection(const MatrixX3d &P, const BasicMesh &mesh) {
  const int nfaces = mesh.NumFaces();
  const int npoints = P.rows();

  // Render the mesh in triangle mode
  const int tex_size = 512;
  OffscreenMeshVisualizer visualizer(tex_size, tex_size);
  visualizer.BindMesh(mesh);
  visualizer.SetFacesToRender(valid_faces);
  visualizer.SetRenderMode(OffscreenMeshVisualizer::Mesh);
  visualizer.SetIndexEncoded(true);
  visualizer.SetMVPMode(OffscreenMeshVisualizer::OrthoNormalExtended);
  QImage img = visualizer.Render();

  visualizer.SetRenderMode(OffscreenMeshVisualizer::BarycentricCoordinates);
  QImage timg = visualizer.Render();

  //cout << "Rendered." << endl;
  //img.save("mesh_projected.png");
  //timg.save("mesh_projected_bcoords.png");

  vector<ICPCorrespondence> corrs(npoints);

  // Draw the points to an image with the same size
  //QImage pimg(tex_size, tex_size, QImage::Format_RGB32);
  //pimg.fill(Qt::black);

  for(int pidx=0;pidx<P.rows();++pidx) {
    double x = P(pidx, 0), y = P(pidx, 1), z = P(pidx, 2);
    ICPCorrespondence bestCorr;

    if(x <= -1 || x >= 1 || y <= -1 || y >= 1) {
      bestCorr.tidx = 0;
      bestCorr.weight = 0;
    } else {
      int px = (1 + x) * 0.5 * tex_size;
      int py = (1 - y) * 0.5 * tex_size;

      QRgb pix = img.pixel(px, py);
      unsigned char r = qRed(pix), g = qGreen(pix), b = qBlue(pix);

      // get triangle index
      int tidx;
      ColorEncoding::decode_index(r, g, b, tidx);

      bestCorr.d = numeric_limits<double>::max();
      bestCorr.tidx = tidx;

      auto face_t = mesh.face(bestCorr.tidx);
      int v0idx = face_t[0], v1idx = face_t[1], v2idx = face_t[2];

      auto v0 = mesh.vertex(v0idx);
      auto v1 = mesh.vertex(v1idx);
      auto v2 = mesh.vertex(v2idx);

      QRgb tpix = timg.pixel(px, py);
      float alpha = qRed(tpix) / 255.0f, beta = qGreen(tpix) / 255.0f, gamma = qBlue(tpix) / 255.0f;

      bestCorr.bcoords[0] = alpha;
      bestCorr.bcoords[1] = beta;
      bestCorr.bcoords[2] = gamma;

      auto hitpoint = alpha * v0 + beta * v1 + gamma * v2;
      bestCorr.hit[0] = hitpoint[0];
      bestCorr.hit[1] = hitpoint[1];
      bestCorr.hit[2] = hitpoint[2];

      double dx = bestCorr.hit[0] - x, dy = bestCorr.hit[1] - y, dz = bestCorr.hit[2] - z;
      bestCorr.d = dx*dx+dy*dy+dz*dz;
      //cout << bestCorr.d << endl;

      // compute face normal
      PhGUtils::Vector3d p0(v0[0], v0[1], v0[2]),
                         p1(v1[0], v1[1], v1[2]),
                         p2(v2[0], v2[1], v2[2]);
      PhGUtils::Vector3d normal(p1-p0, p2-p0);
      PhGUtils::Vector3d dvec(dx, dy, dz);
      double scaler = 1-1/(1+exp(-(250*((bestCorr.d<0?bestCorr.d:sqrt(bestCorr.d))-0.025))));
      bestCorr.weight = fabs(normal.normalized().dot(dvec.normalized())) * scaler;

      //pimg.setPixel(px, py, qRgb(255*scaler, 0, 255*(1-scaler)));
    }

    corrs[pidx] = bestCorr;
  }

  //pimg.save("points_projected.png");
  //exit(0);

  return corrs;
}

vector<ICPCorrespondence> MeshDeformer::findClosestPoints_tree(const MatrixX3d &P, const BasicMesh &mesh) {
  int nfaces = mesh.NumFaces();
  int npoints = P.rows();

  PhGUtils::Timer t_buildtree, t_treesearch, t_bary;

  t_buildtree.tic();
  vector<int> face_indices_map;
  std::vector<Triangle> triangles;
  triangles.reserve(nfaces);
  if(valid_faces.empty()) {
    cout << "Using all faces ..." << endl;
    face_indices_map.resize(nfaces);
    for(int i=0,ioffset=0;i<nfaces;++i) {
      face_indices_map[i] = i;
      auto face_i = mesh.face(i);
      int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
      auto p1 = mesh.vertex(v1), p2 = mesh.vertex(v2), p3 = mesh.vertex(v3);
      Point a(p1[0], p1[1], p1[2]);
      Point b(p2[0], p2[1], p2[2]);
      Point c(p3[0], p3[1], p3[2]);

      triangles.push_back(Triangle(a, b, c));
    }
  } else {
    face_indices_map.resize(valid_faces.size());
    int idx = 0;
    for(int i : valid_faces) {
      auto face_i = mesh.face(i);
      face_indices_map[idx++] = i;
      int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
      auto p1 = mesh.vertex(v1), p2 = mesh.vertex(v2), p3 = mesh.vertex(v3);
      Point a(p1[0], p1[1], p1[2]);
      Point b(p2[0], p2[1], p2[2]);
      Point c(p3[0], p3[1], p3[2]);

      triangles.push_back(Triangle(a, b, c));
    }
  }

  Tree tree(triangles.begin(), triangles.end());
  tree.accelerate_distance_queries();
  t_buildtree.toc();

  vector<ICPCorrespondence> corrs(npoints);

  // query the tree for closest point
  //#pragma omp parallel for
  for (int pidx = 0; pidx < npoints; ++pidx) {
    int poffset = pidx * 3;
    double px = P(pidx, 0), py = P(pidx, 1), pz = P(pidx, 2);

#undef max
    ICPCorrespondence bestCorr;
    bestCorr.d = numeric_limits<double>::max();
    t_treesearch.tic();
    Tree::Point_and_primitive_id bestHit = tree.closest_point_and_primitive(Point(px, py, pz));
    t_treesearch.toc();

    bestCorr.tidx = face_indices_map[bestHit.second - triangles.begin()];
    bestCorr.hit[0] = bestHit.first.x(); bestCorr.hit[1] = bestHit.first.y(); bestCorr.hit[2] = bestHit.first.z();
    double dx = px - bestHit.first.x(), dy = py - bestHit.first.y(), dz = pz - bestHit.first.z();
    bestCorr.d = dx*dx+dy*dy+dz*dz;

    // compute bary-centric coordinates
    int toffset = bestCorr.tidx * 3;
    auto face_t = mesh.face(bestCorr.tidx);
    int v0idx = face_t[0], v1idx = face_t[1], v2idx = face_t[2];

    auto v0 = mesh.vertex(v0idx);
    auto v1 = mesh.vertex(v1idx);
    auto v2 = mesh.vertex(v2idx);

    PhGUtils::Point3f bcoords;
    t_bary.tic();
    PhGUtils::computeBarycentricCoordinates(PhGUtils::Point3f(px, py, pz),
                                            PhGUtils::Point3f(v0[0], v0[1], v0[2]),
                                            PhGUtils::Point3f(v1[0], v1[1], v1[2]),
                                            PhGUtils::Point3f(v2[0], v2[1], v2[2]),
                                            bcoords);
    t_bary.toc();
    bestCorr.bcoords[0] = bcoords.x; bestCorr.bcoords[1] = bcoords.y; bestCorr.bcoords[2] = bcoords.z;

    // compute face normal
    PhGUtils::Vector3d p0(v0[0], v0[1], v0[2]),
                       p1(v1[0], v1[1], v1[2]),
                       p2(v2[0], v2[1], v2[2]);
    PhGUtils::Vector3d normal(p1-p0, p2-p0);
    PhGUtils::Vector3d dvec(dx, dy, dz);
    bestCorr.weight = fabs(normal.normalized().dot(dvec.normalized()));

    corrs[pidx] = bestCorr;
  }

  t_buildtree.report("building search tree");
  t_treesearch.report("searching in tree");
  t_bary.report("computing barycentric coordinates");

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
