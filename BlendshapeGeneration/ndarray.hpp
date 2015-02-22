#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "common.h"

template <typename T>
struct Array2D {
  Array2D():nrow(0), ncol(0), data(shared_ptr<T>(nullptr)){}
  Array2D(int m, int n):nrow(m), ncol(n), data(shared_ptr<T>(new T[m*n])){}
  Array2D clone() const {
    Array2D ret(nrow, ncol);
    memcpy(ret.data.get(), data.get(), sizeof(T)*nrow*ncol);
    return ret;
  }

  T& operator()(int r, int c) { return data.get()[r*ncol+c]; }
  const T& operator()(int r, int c) const { return data.get()[r*ncol+c]; }

  T& operator()(int idx) { return data.get()[idx]; }
  const T& operator()(int idx) const { return data.get()[idx]; }

  Array2D row(const vector<int> &rowIndices) {
    Array2D ret(rowIndices.size(), ncol);
    for(int i=0;i<rowIndices.size();++i) {
      for(int j=0;j<ncol;++j) ret(i, j) = (*this)(rowIndices[i], j);
    }
    return ret;
  }

  void resize(int m, int n) {
    nrow = m; ncol = n;
    data.reset(new T[m*n]);
  }

  T* rowptr(int ridx) const{
    return data.get() + ridx*ncol;
  }

  void save(const string &filename) {
    ofstream fout(filename);
    fout << (*this);
    fout.close();
  }

  template <typename AT>
  friend ostream& operator<<(ostream &os, const Array2D<AT> &A);

  int nrow, ncol;
  shared_ptr<T> data;
};

template <typename T>
ostream &operator<<(ostream &os, const Array2D<T> &A) {
  for(int i=0;i<A.nrow;++i) {
    for(int j=0;j<A.ncol;++j) {
      os << A(i, j) << ' ';
    }
    os << endl;
  }
  return os;
}

#endif // NDARRAY_HPP

