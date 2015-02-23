#pragma once

#include "ndarray.hpp"

struct PointCloud
{
  PointCloud();
  ~PointCloud();

  PointCloud(const PointCloud& other);
  PointCloud(PointCloud &&other);
  PointCloud& operator=(const PointCloud &rhs);
  PointCloud& operator=(PointCloud &&rhs);

  void write(const string &filename);

  friend ostream& operator<<(ostream &os, const PointCloud &P);
  Array2D<double> points;
};


inline ostream& operator<<(ostream &os, const PointCloud &P)
{
  os << P.points;
  return os;
}

