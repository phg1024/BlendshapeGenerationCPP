#ifndef TRIANGLE_GRADIENT_H
#define TRIANGLE_GRADIENT_H

#include "Geometry/matrix.hpp"
#include <MultilinearReconstruction/basicmesh.h>

inline PhGUtils::Matrix3x3d triangleGradient(const BasicMesh &m, int fidx) {
  auto f = m.face(fidx);

  Vector3d v0 = m.vertex(f[0]);
  Vector3d v1 = m.vertex(f[1]);
  Vector3d v2 = m.vertex(f[2]);

  Vector3d v1mv0 = v1 - v0;
  Vector3d v2mv0 = v2 - v0;

  PhGUtils::Vector3d v0v1(v1mv0[0], v1mv0[1], v1mv0[2]);
  PhGUtils::Vector3d v0v2(v2mv0[0], v2mv0[1], v2mv0[2]);

  PhGUtils::Vector3d n(v0v1, v0v2);
  PhGUtils::Vector3d nn = n.normalized();

  PhGUtils::Matrix3x3d G(v0v1, v0v2, nn);

  double d = 0.5 * dot(n, nn);

  return G;
}

inline pair<PhGUtils::Matrix3x3d, double> triangleGradient2(const BasicMesh &m, int fidx) {
  auto f = m.face(fidx);

  Vector3d v0 = m.vertex(f[0]);
  Vector3d v1 = m.vertex(f[1]);
  Vector3d v2 = m.vertex(f[2]);

  Vector3d v1mv0 = v1 - v0;
  Vector3d v2mv0 = v2 - v0;

  PhGUtils::Vector3d v0v1(v1mv0[0], v1mv0[1], v1mv0[2]);
  PhGUtils::Vector3d v0v2(v2mv0[0], v2mv0[1], v2mv0[2]);

  PhGUtils::Vector3d n(v0v1, v0v2);
  PhGUtils::Vector3d nn = n.normalized();

  PhGUtils::Matrix3x3d G(v0v1, v0v2, nn);

  double d = 0.5 * n.dot(nn);

  return make_pair(G, d);
}
#endif // TRIANGLE_GRADIENT_H

