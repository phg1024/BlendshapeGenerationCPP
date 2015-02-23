#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "common.h"

template <typename T>
struct Array1D {
  Array1D():nrow(0), data(shared_ptr<T>(nullptr)){}
  Array1D(int n):nrow(n), data(shared_ptr<T>(new T[n])){}
  Array1D(const Array1D &other):nrow(other.nrow) {
    data = shared_ptr<T>(new T[nrow]);
    memcpy(data.get(), other.data.get(), sizeof(T)*nrow);
  }

//  Array1D(Array1D &&other):nrow(other.nrow) {
//    data = other.data;
//    other.data.reset();
//  }
  Array1D clone() const {
    Array1D ret(nrow);
    memcpy(ret.data.get(), data.get(), sizeof(T)*nrow);
    return ret;
  }
  Array1D &operator=(const Array1D &rhs) {
    if( this != &rhs) {
      nrow = rhs.nrow;
      data = shared_ptr<T>(new T[nrow]);
      memcpy(data.get(), rhs.data.get(), sizeof(T)*nrow);
    }
    return (*this);
  }
//  Array1D &operator=(Array1D &&rhs) {
//    if( this != &rhs) {
//      nrow = rhs.nrow;
//      data = rhs.data;
//      rhs.data.reset();
//    }
//    return (*this);
//  }
  static Array1D zeros(int n) {
    Array1D ret(n);
    memset(ret.data.get(), 0, sizeof(T)*n);
    return ret;
  }

  static Array1D random(int n) {
    Array1D ret(n);
    for(int i=0;i<n;++i) ret(i)=rand()/(double)RAND_MAX;
    return ret;
  }

  void resize(int n) {
    nrow = n;
    data.reset(new T[n]);
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
//  Array2D(Array2D &&other):nrow(other.nrow), ncol(other.ncol) {
//    data = other.data;
//    other.data.reset();
//  }

  Array2D clone() const {
    Array2D ret(nrow, ncol);
    memcpy(ret.data.get(), data.get(), sizeof(T)*nrow*ncol);
    return ret;
  }

  Array2D& operator=(const Array2D& rhs) {
    if( this != &rhs ) {
      nrow = rhs.nrow;
      ncol = rhs.ncol;
      data = shared_ptr<T>(new T[nrow*ncol]);
      memcpy(data.get(), rhs.data.get(), sizeof(T)*nrow*ncol);
    }
    return (*this);
  }

//  Array2D& operator=(Array2D&& rhs) {
//    if( this != &rhs ) {
//      nrow = rhs.nrow;
//      ncol = rhs.ncol;
//      data = rhs.data;
//      rhs.data.reset();
//    }
//    return (*this);
//  }

  static Array2D zeros(int m, int n) {
    Array2D ret(m, n);
    memset(ret.data.get(), 0, sizeof(T)*m*n);
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

  Array2D& operator+=(const Array2D &rhs) {
    if( nrow != rhs.nrow || ncol != rhs.ncol ) {
      throw "array dimension does not match";
    }
    else {
      int n = nrow * ncol;
      for(int i=0;i<n;++i) {
        data.get()[i] += rhs(i);
      }
      return (*this);
    }
  }

  Array2D operator-(const Array2D &rhs) {
    if( nrow != rhs.nrow || ncol != rhs.ncol ) {
      throw "array dimension does not match";
    }
    else {
      Array2D res((*this));
      int n = nrow * ncol;
      for(int i=0;i<n;++i) {
        res(i) -= rhs(i);
      }
      return res;
    }
  }

  template <typename AT>
  friend Array2D<AT> operator*(AT lhs, const Array2D<AT> &rhs);

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

template <typename T>
Array2D<T> operator*(T lhs, const Array2D<T> &rhs) {
  Array2D<T> ret = rhs;
  int n = rhs.nrow * rhs.ncol;
  for(int i=0;i<n;++i) ret(i) *= lhs;
  return ret;
}

#endif // NDARRAY_HPP

