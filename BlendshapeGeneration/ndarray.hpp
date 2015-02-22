#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "common.h"

template <typename T>
struct Array1D {
  Array1D():nrow(0), data(shared_ptr<T>(nullptr)){}
  Array1D(int n):nrow(n), data(shared_ptr<T>(new T[n])){}
  Array1D(const Array1D &other):nrow(other.nrow) {
    data = shared_ptr<T>(new T[nrow]);
    memcpy(data.get(), other.data.get(), sizeof(T)*nrow*ncol);
  }
  Array1D clone() const {
    Array1D ret(nrow);
    memcpy(ret.data.get(), data.get(), sizeof(T)*nrow);
    return ret;
  }

  T& operator()(int idx) { return data.get()[idx]; }
  const T& operator()(int idx) const { return data.get()[idx]; }

  int nrow;
  shared_ptr<T> data;
};

template <typename T>
struct Array2D {
  Array2D():nrow(0), ncol(0), data(shared_ptr<T>(nullptr)){}
  Array2D(int m, int n):nrow(m), ncol(n), data(shared_ptr<T>(new T[m*n])){}
  Array2D(const Array2D &other):nrow(other.nrow), ncol(other.ncol) {
    data = shared_ptr<T>(new T[other.nrow*other.ncol]);
    memcpy(data.get(), other.data.get(), sizeof(T)*nrow*ncol);
  }

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

  Array2D operator-(const Array2D &rhs) {
    if( nrow != rhs.nrow || ncol != rhs.ncol ) {
      throw "array dimension does not match";
    }
    else {
      Array2D res((*this));
      int n = nrow * ncol;
      for(int i=0;i<n;++i) {
        res[i] -= rhs(i);
      }
      return res;
    }
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

