#pragma once
#include "ndarray.hpp"

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

  void load(const string &filename);
  void write(const string &filename);

  Array2D<int> faces;
  Array2D<double> verts;
};

