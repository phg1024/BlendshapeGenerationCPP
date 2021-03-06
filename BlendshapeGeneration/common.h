#pragma once

#define USE_BOOST 0

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <vector>
#include <list>
#include <queue>
#include <map>
#if USE_BOOST
#include <boost/unordered_map.hpp>
#else
#include <unordered_map>
#endif
#include <set>
#if USE_BOOST
#include <boost/unordered_set.hpp>
#else
#include <unordered_set>
#endif
#include <string>
#include <cstring>
#include <memory>
#if USE_BOOST
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
#endif
#include <thread>
#include <chrono>

using namespace std;

#include "Utils/Timer.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
//using namespace Eigen;
