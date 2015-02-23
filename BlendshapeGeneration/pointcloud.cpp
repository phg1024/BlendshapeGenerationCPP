#include "pointcloud.h"


PointCloud::PointCloud()
{
}


PointCloud::~PointCloud()
{
}

PointCloud::PointCloud(const PointCloud &other)
{
  points = other.points;
}

PointCloud::PointCloud(PointCloud &&other)
{
  points = std::move(other.points);
}

PointCloud &PointCloud::operator=(const PointCloud &rhs)
{
  if( this != &rhs ) {
    points = rhs.points;
  }
  return (*this);
}

PointCloud &PointCloud::operator=(PointCloud &&rhs)
{
  if( this != &rhs ) {
    points = std::move(rhs.points);
  }
  return (*this);
}
