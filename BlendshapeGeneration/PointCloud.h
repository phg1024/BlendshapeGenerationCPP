#pragma once

#include "BasicMatrix.hpp"

struct PointCloud
{
  PointCloud();
  ~PointCloud();

  friend ostream& operator<<(ostream &os, const PointCloud &P);
  BasicMatrix<double> points;
};


inline ostream& operator<<(ostream &os, const PointCloud &P)
{
  os << P.points;
  return os;
}

