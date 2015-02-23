#ifndef UTILS_H
#define UTILS_H

#include "Geometry/matrix.hpp"
#include "basicmesh.h"

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

inline PhGUtils::Matrix3x3d triangleGradient(const BasicMesh &m, int fidx) {
  int* v = m.faces.rowptr(fidx);

  PhGUtils::Vector3d v0(m.verts.rowptr(v[0]));
  PhGUtils::Vector3d v1(m.verts.rowptr(v[1]));
  PhGUtils::Vector3d v2(m.verts.rowptr(v[2]));

  PhGUtils::Vector3d v0v1 = v1 - v0;
  PhGUtils::Vector3d v0v2 = v2 - v0;

  PhGUtils::Vector3d n(v0v1, v0v2);
  PhGUtils::Vector3d nn = n.normalized();

  PhGUtils::Matrix3x3d G(v0v1, v0v2, nn);

  double d = 0.5 * dot(n, nn);

  return G;
}

inline pair<PhGUtils::Matrix3x3d, double> triangleGradient2(const BasicMesh &m, int fidx) {
  int *v = m.faces.rowptr(fidx);

  PhGUtils::Vector3d v0(m.verts.rowptr(v[0]));
  PhGUtils::Vector3d v1(m.verts.rowptr(v[1]));
  PhGUtils::Vector3d v2(m.verts.rowptr(v[2]));

  PhGUtils::Vector3d v0v1 = v1 - v0;
  PhGUtils::Vector3d v0v2 = v2 - v0;

  PhGUtils::Vector3d n(v0v1, v0v2);
  PhGUtils::Vector3d nn = n.normalized();

  PhGUtils::Matrix3x3d G(v0v1, v0v2, nn);

  double d = 0.5 * n.dot(nn);

  return make_pair(G, d);
}
#endif // UTILS_H

