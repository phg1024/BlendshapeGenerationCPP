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

#include "MultilinearReconstruction/eigen_headers.h"
