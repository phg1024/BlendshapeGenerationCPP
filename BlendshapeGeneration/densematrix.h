#pragma once
#include "common.h"
#include "cholmod.h"

#define USE_MKL 1
#if USE_MKL
#include "mkl.h"
#else
#include "Accelerate/Accelerate.h"
#endif

class DenseMatrix {
public:
  DenseMatrix();
  DenseMatrix(int m, int n);
  DenseMatrix(const DenseMatrix &other);
  DenseMatrix(DenseMatrix &&other);
  DenseMatrix& operator=(const DenseMatrix &other);
  DenseMatrix& operator=(DenseMatrix &&other);

  static DenseMatrix zeros(int m, int n);

  static DenseMatrix random(int m, int n);

  static DenseMatrix eye(int n) {
    DenseMatrix M = DenseMatrix::zeros(n, n);
    for (int i = 0; i < n; ++i) M(i*n+i) = 1.0;
    return M;
  }

  int nrow() const { return ptr->nrow; }
  int ncol() const { return ptr->ncol; }

  double& operator()(int idx) {
    return x[idx];
  }
  const double& operator()(int idx) const{
    return x[idx];
  }

  double& operator()(int ridx, int cidx) {
    return x[cidx*ptr->nrow + ridx];
  }

  const double& operator()(int ridx, int cidx) const {
    return x[cidx*ptr->nrow + ridx];
  }

  void resize(int m, int n);

  DenseMatrix& operator*=(double s);

  DenseMatrix operator+(const DenseMatrix &rhs);

  DenseMatrix operator*(const DenseMatrix &rhs);

  DenseMatrix transposed() const;

  DenseMatrix inv() const;

  friend ostream& operator<<(ostream &os, const DenseMatrix &P);

private:
  cholmod_dense *ptr;
  double *x;
  void resetPointers();
};

inline ostream& operator<<(ostream &os, const DenseMatrix &M) {
  for (int i = 0; i < M.nrow(); ++i) {
    for (int j = 0; j < M.ncol(); ++j) {
      cout << M(i, j) << ' ';
    }
    cout << endl;
  }
  return os;
}
