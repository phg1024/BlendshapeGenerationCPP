#pragma once
#include "common.h"

#include "mkl.h"

template <typename T>
struct BasicMatrix {
  BasicMatrix():rows(0),cols(0){}
  BasicMatrix(const BasicMatrix &other){
    rows = other.rows;
    cols = other.cols;
    data = other.data;
  }
  BasicMatrix& operator=(const BasicMatrix &other) {
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    return *this;
  }
  BasicMatrix(int m, int n):rows(m),cols(n),data(shared_ptr<T>(new T[m*n])) {}

  BasicMatrix clone() const {
    BasicMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = shared_ptr<T>(new T[rows*cols]);
    memcpy(m.data.get(), data.get(), sizeof(T)*rows*cols);
    return m;
  }

  static BasicMatrix zeros(int m, int n) {
    BasicMatrix M(m, n);
    memset(M.data.get(), 0, sizeof(T)*m*n);
    return M;
  }

  static BasicMatrix random(int m, int n) {
    BasicMatrix M(m, n);
    for (int i = 0; i < m*n; ++i) M(i) = rand() / (T)RAND_MAX;
    return M;
  }

  static BasicMatrix eye(int n) {
    BasicMatrix M = BasicMatrix::zeros(n, n);
    for (int i = 0; i < n; ++i) M(i*n+i) = 1.0;
    return M;
  }


  T& operator()(int idx) {
    return data.get()[idx];
  }
  const T& operator()(int idx) const{
    return data.get()[idx];
  }

  T& operator()(int ridx, int cidx) {
    return data.get()[ridx*cols + cidx];
  }

  const T& operator()(int ridx, int cidx) const {
    return data.get()[ridx*cols + cidx];
  }

  void resize(int m, int n) {
    rows = m; cols = n;
    data.reset(new T[m*n]);
  }
  BasicMatrix row(const vector<int> &rowIndices) {
    BasicMatrix R(rowIndices.size(), cols);
    T *dptr = data.get();
    for (int i = 0, offset=0; i < rowIndices.size(); ++i) {
      for (int j = 0,doffset=rowIndices[i]*cols; j < cols; ++j) {
        R(offset++) = dptr[doffset++];
      }        
    }
    return R;
  }

  const T* rowptr(int ridx) const {
    return data.get() + ridx*cols;
  }

  BasicMatrix& operator*=(T s) {
    cblas_sscal(rows*cols, s, data.get(), 1);
    return (*this);
  }

  BasicMatrix operator+(const BasicMatrix &rhs) {
    if (rows != rhs.rows || cols != rhs.cols) throw "Matrix dimensions not compatible.";
    else {
      BasicMatrix res = rhs.clone();
      cblas_saxpy(rows*cols, 1.0, data.get(), 1, res.data.get(), 1);
      return res;
    }
  }

  BasicMatrix operator*(const BasicMatrix &rhs) {
    if (cols != rhs.rows) throw "Matrix dimensions not compatible.";
    else {
      BasicMatrix res(rows, rhs.cols);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, rhs.cols, cols, 1.0, data.get(), cols,
        rhs.data.get(), rhs.cols, 0, res.data.get(), rhs.cols);
      return res;
    }
  }

  BasicMatrix transposed() const {
    BasicMatrix res(cols, rows);
    for (int i = 0,idx=0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        res(j*rows+i) = data.get()[idx]; ++idx;
      }
    }
    return res;
  }

  BasicMatrix inv() const {
    if (rows != cols) throw "Not a square matrix!";
    BasicMatrix res = this->clone();
    vector<int> ipiv(rows);
    // LU factorization
    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, rows, cols, res.data.get(), cols, &ipiv[0]);
    // inversion
    LAPACKE_sgetri(LAPACK_ROW_MAJOR, rows, res.data.get(), cols, &ipiv[0]);
    return res;
  }
  template <typename MT>
  friend ostream& operator<<(ostream &os, const BasicMatrix<MT> &P);
  int rows, cols;
  shared_ptr<T> data;
};

template <typename T>
ostream& operator<<(ostream &os, const BasicMatrix<T> &M) {
  for (int i = 0,offset=0; i < M.rows; ++i) {
    for (int j = 0; j < M.cols; ++j) {
      cout << M(offset) << ' '; ++offset;
    }
    cout << endl;
  }
  return os;
}