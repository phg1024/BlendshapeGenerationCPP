#pragma once
#include "common.h"
#include "BasicMesh.h"
#include "PointCloud.h"
#include "SparseMatrix.hpp"

struct ICPCorrespondence {
  int tidx;             // triangle index
  float bcoords[3];     // bary-centric coordinates
  float hit[3];         // point on triangle
  float d;
  float weight;         // weight of this point
};

class MeshDeformer
{
public:
  MeshDeformer();
  ~MeshDeformer();
  
  void setSource(const BasicMesh &src) { S = src; }
  void setLandmarks(const vector<int> &lms) { landmarks = lms; }
  BasicMesh deformWithMesh(const BasicMesh &T, const PointCloud &lm_points);
  BasicMesh deformWithPoints(const PointCloud &P, const PointCloud &lm_points);

protected:
  vector<ICPCorrespondence> findClosestPoints_bruteforce(const PointCloud &P, const BasicMesh &mesh);
  ICPCorrespondence findClosestPoint_triangle(float px, float py, float pz,
                                              const float *v0, const float *v1, const float *v2);

private:
  BasicMesh S;
  vector<int> landmarks;
};

