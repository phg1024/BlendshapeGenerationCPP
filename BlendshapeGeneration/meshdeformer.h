#pragma once
#include "common.h"
#include <MultilinearReconstruction/basicmesh.h>
#include "pointcloud.h"
//#include "sparsematrix.h"

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
};
