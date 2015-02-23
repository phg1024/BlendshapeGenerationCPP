#include "densematrix.h"

DenseMatrix::DenseMatrix():ptr(nullptr), x(nullptr){}

DenseMatrix::~DenseMatrix()
{
  if( ptr != nullptr ) {
    // FIXME, memory issue.
    cholmod_free_dense(&ptr, global::cm);
    ptr = nullptr;
    x = nullptr;
  }
}

DenseMatrix::DenseMatrix(int m, int n)
{
  ptr = cholmod_allocate_dense(m, n, m, CHOLMOD_REAL, global::cm);
  resetPointers();
}

DenseMatrix::DenseMatrix(const DenseMatrix &other){
  ptr = cholmod_copy_dense(other.ptr, global::cm);
  resetPointers();
}

DenseMatrix::DenseMatrix(DenseMatrix &&other):ptr(nullptr){
  ptr = other.ptr;
  resetPointers();
  other.ptr = nullptr;
}

DenseMatrix &DenseMatrix::operator=(const DenseMatrix &other) {
  if( this != &other ) {
    if( ptr != nullptr ) cholmod_free_dense(&ptr, global::cm);
    ptr = cholmod_copy_dense(other.ptr, global::cm);
    resetPointers();
  }
  return *this;
}

DenseMatrix &DenseMatrix::operator=(DenseMatrix &&other) {
  if( this != &other ) {
    ptr = other.ptr;
    resetPointers();
    other.ptr = nullptr;
  }
  return *this;
}

DenseMatrix DenseMatrix::zeros(int m, int n) {
  DenseMatrix M(m, n);
  memset(M.x, 0, sizeof(double)*m*n);
  return M;
}

DenseMatrix DenseMatrix::random(int m, int n) {
  DenseMatrix M(m, n);
  for (int i = 0; i < m*n; ++i) M(i) = rand() / (double)RAND_MAX;
  return M;
}

void DenseMatrix::resize(int m, int n) {
  if( ptr != nullptr ) cholmod_free_dense(&ptr, global::cm);
  ptr = cholmod_allocate_dense(m, n, m, CHOLMOD_REAL, global::cm);
  resetPointers();
}

void DenseMatrix::resetPointers() {
  if( ptr != nullptr ) x = (double*) (ptr->x);
}

DenseMatrix &DenseMatrix::operator*=(double s) {
  cblas_dscal(ptr->nrow*ptr->ncol, s, x, 1);
  return (*this);
}

DenseMatrix DenseMatrix::operator+(const DenseMatrix &rhs) {
  if (ptr->nrow != rhs.nrow() || ptr->ncol != rhs.ncol()) throw "Matrix dimensions not compatible.";
  else {
    DenseMatrix res = rhs;
    cblas_daxpy(ptr->nrow*ptr->ncol, 1.0, x, 1, res.x, 1);
    return res;
  }
}

DenseMatrix DenseMatrix::operator*(const DenseMatrix &rhs) {
  if (ptr->ncol != rhs.nrow()) throw "Matrix dimensions not compatible.";
  else {
    DenseMatrix res(ptr->nrow, rhs.ncol());

#if 0
    int m = ptr->nrow, n = rhs.ptr->ncol, k = ptr->ncol;

    const double *A = (const double*)(ptr->x), *B = (const double*)(rhs.ptr->x);
    double *C = (double*)(res.ptr->x);

    double alpha = 1.0, beta = 1.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);

#else
    for(int i=0;i<ptr->nrow;++i) {
      for(int j=0;j<rhs.ncol();++j) {
        double sum = 0;
        for(int k=0;k<ptr->ncol;++k) {
          sum += (*this)(i, k) * rhs(k, j);
        }
        res(i, j) = sum;
      }
    }
#endif
    return res;
  }
}

DenseMatrix DenseMatrix::transposed() const {
  DenseMatrix res(ptr->ncol, ptr->nrow);
  for (int i = 0; i < ptr->ncol; ++i) {
    for (int j = 0; j < ptr->nrow; ++j) {
      res(i, j) = (*this)(j, i);
    }
  }
  return res;
}

DenseMatrix DenseMatrix::inv() const {
  if (ptr->nrow!= ptr->ncol) throw "Not a square matrix!";
  DenseMatrix res = (*this);
  vector<int> ipiv(ptr->nrow*2);
  int m = res.nrow(), n = res.ncol();
  double *A = (double*)(res.ptr->x);
#if USE_MKL
  // LU factorization
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, A, m, &(ipiv[0]));
  // inversion
  LAPACKE_dgetri(LAPACK_COL_MAJOR, m, A, m, &(ipiv[0]));
#else
  int info;
  dgetrf_(&m, &n, A, &m, &(ipiv[0]), &info);
  dgetri_(&m, A, &m, &(ipiv[0]), 0, 0, 0);
#endif
  return res;
}


