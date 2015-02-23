#pragma once
#include "ndarray.hpp"
#include "pointcloud.h"

#include "Geometry/MeshLoader.h"

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

struct BasicMesh
{
  BasicMesh();
  BasicMesh(const BasicMesh &other) {
    faces = other.faces;
    verts = other.verts;
  }
  ~BasicMesh();

  BasicMesh& operator=(const BasicMesh &other) {
    if (this != &other) {
      faces = other.faces;
      verts = other.verts;
    }
    return *this;
  }

  BasicMesh clone() const {
    BasicMesh m;
    m.faces = faces.clone();
    m.verts = verts.clone();
    return m;
  }

  PointCloud samplePoints(int points_per_face, double zcutoff) const;
  template <typename Pred>
  vector<int> filterFaces(Pred p);
  template <typename Pred>
  vector<int> filterVertices(Pred p);

  void load(const string &filename);
  void write(const string &filename);

  Array2D<int> faces;
  Array2D<double> verts;
};

template <typename Pred>
vector<int> BasicMesh::filterFaces(Pred p)
{
  vector<int> v;
  for(int i=0;i<faces.nrow;++i) {
    if( p(faces.rowptr(i)) ) {
      v.push_back(i);
    }
  }
  return v;
}

template <typename Pred>
vector<int> BasicMesh::filterVertices(Pred p)
{
  vector<int> v;
  for(int i=0;i<verts.nrow;++i) {
    if( p(verts.rowptr(i)) ) {
      v.push_back(i);
    }
  }
  return v;
}

