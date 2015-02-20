#pragma once

#include "ndarray.hpp"

struct PointCloud
{
  PointCloud();
  ~PointCloud();

  friend ostream& operator<<(ostream &os, const PointCloud &P);
  Array2D<double> points;
};


inline ostream& operator<<(ostream &os, const PointCloud &P)
{
  os << P.points;
  return os;
}

