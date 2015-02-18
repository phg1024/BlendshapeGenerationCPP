#pragma once

#include "BasicMatrix.hpp"

struct PointCloud
{
  PointCloud();
  ~PointCloud();

  friend ostream& operator<<(ostream &os, const PointCloud &P);
  BasicMatrix<float> points;
};


inline ostream& operator<<(ostream &os, const PointCloud &P)
{
  os << P.points;
  return os;
}

