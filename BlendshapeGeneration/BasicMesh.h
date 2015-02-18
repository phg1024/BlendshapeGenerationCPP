#pragma once
#include "BasicMatrix.hpp"

#include "Geometry/MeshLoader.h"

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

  BasicMatrix<int> faces;
  BasicMatrix<float> verts;
};

