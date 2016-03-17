#pragma once
#include "common.h"
#include <MultilinearReconstruction/basicmesh.h>
#include "pointcloud.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
using namespace Eigen;

struct ICPCorrespondence {
  int tidx;             // triangle index
  double bcoords[3];     // bary-centric coordinates
  double hit[3];         // point on triangle
  double d;
  double weight;         // weight of this point
};

class MeshDeformer
{
public:
  MeshDeformer();
  ~MeshDeformer();

  void setSource(const BasicMesh &src) { S = src; }
  void setLandmarks(const vector<int> &lms) { landmarks = lms; }
  void setValidFaces(const vector<int> &fidx) {
    valid_faces = fidx;

    valid_vertices.clear();
    unordered_set<int> valid_vertices_set;
    for(int i : valid_faces) {
      auto fi = S.face(i);
      valid_vertices_set.insert(fi[0]);
      valid_vertices_set.insert(fi[1]);
      valid_vertices_set.insert(fi[2]);
    }
    valid_vertices.assign(valid_vertices_set.begin(), valid_vertices_set.end());

    // Establish vertex-index mapping
    vertex_index_map.clear();
    vertex_index_map.resize(S.NumVertices(), -1);
    for(int i = 0; i<valid_vertices.size();++i) {
      vertex_index_map[valid_vertices[i]] = i;
    }

    #if 0
    {
      ofstream fout("valid_faces.txt");
      for(int i : fidx) {
        fout << i << endl;
      }
      fout.close();
    }
    #endif

    // Set the fixed faces
    fixed_faces.clear();
    unordered_set<int> valid_faces_set(valid_faces.begin(), valid_faces.end());
    for(int i=0;i<S.NumFaces();++i) {
      if(valid_faces_set.count(i) == 0) fixed_faces.push_back(i);
    }

    // Collect the fixed vertices
    fixed_vertices.clear();
    unordered_set<int> fixed_vertices_set;

    struct edge_t {
      edge_t() {}
      edge_t(int s, int t) : s(s), t(t) {}
      bool operator<(const edge_t& other) const {
        if(s < other.s) return true;
        else if( s > other.s ) return false;
        else return t < other.t;
      }
      int s, t;
    };
    map<edge_t, int> counter;
    for(int i : fixed_faces) {
      auto fi = S.face(i);
      fixed_vertices_set.insert(fi[0]);
      fixed_vertices_set.insert(fi[1]);
      fixed_vertices_set.insert(fi[2]);

      auto add_edge = [&counter](int s, int t) {
        edge_t e(min(s, t), max(s, t));
        if(counter.count(e)) ++counter[e];
        else counter[e] = 1;
      };

      add_edge(fi[0], fi[1]);
      add_edge(fi[1], fi[2]);
      add_edge(fi[2], fi[0]);
    }
    fixed_vertices.assign(fixed_vertices_set.begin(), fixed_vertices_set.end());

    // Find out the boundary vertices for the fixed region
    unordered_set<int> boundary_vertices_set;
    fixed_faces_boundary_vertices.clear();
    for(auto p : counter) {
      if(p.second == 1) {
        boundary_vertices_set.insert(p.first.s);
        boundary_vertices_set.insert(p.first.t);
      }
    }
    fixed_faces_boundary_vertices.assign(boundary_vertices_set.begin(), boundary_vertices_set.end());

    #if 0
    {
      ofstream fout("boundary_vertices.txt");
      for(int i : fixed_faces_boundary_vertices) {
        fout << i << endl;
      }
      fout.close();
    }
    #endif
  }

  BasicMesh deformWithMesh(const BasicMesh &T, const PointCloud &lm_points, int itmax = 10);
  BasicMesh deformWithPoints(const MatrixX3d &P, const PointCloud &lm_points, int itmax = 10);

protected:
  vector<ICPCorrespondence> findClosestPoints_tree(const MatrixX3d &P, const BasicMesh &mesh);
  vector<ICPCorrespondence> findClosestPoints_bruteforce(const MatrixX3d &P, const BasicMesh &mesh);
  ICPCorrespondence findClosestPoint_triangle(double px, double py, double pz,
                                              const Vector3d& v0, const Vector3d& v1, const Vector3d& v2);

private:
  BasicMesh S;
  vector<int> landmarks;

  // The set of vertices/faces to deform
  vector<int> valid_vertices;
  vector<int> valid_faces;
  vector<int> vertex_index_map;

  // The set of fixed vertices/faces
  vector<int> fixed_faces;
  vector<int> fixed_vertices;
  vector<int> fixed_faces_boundary_vertices;
};
