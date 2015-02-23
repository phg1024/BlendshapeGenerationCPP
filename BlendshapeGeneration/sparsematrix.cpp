#include "sparsematrix.h"


SparseMatrix::~SparseMatrix()
{
//  if( ptr != nullptr ) {
//    cholmod_free_triplet(&ptr, global::cm);
//  }
}

SparseMatrix::SparseMatrix(int m, int n, int nzmax)
{
  ptr = cholmod_allocate_triplet(m, n, nzmax, 0, CHOLMOD_REAL, global::cm);
  resetPointers();
}

SparseMatrix::SparseMatrix(const SparseMatrix &other)
{
  ptr = cholmod_copy_triplet(other.ptr, global::cm);
  resetPointers();
}

SparseMatrix &SparseMatrix::operator=(const SparseMatrix &rhs)
{
  if( this != &rhs ) {
    if( ptr != nullptr ) cholmod_free_triplet(&ptr, global::cm);
    ptr = cholmod_copy_triplet(rhs.ptr, global::cm);
  }
  return (*this);
}

void SparseMatrix::append(int i, int j, double v)
{
  if( ptr->nnz +1 > ptr->nzmax ) {
    cout << "need resize ..." << endl;
    resize(ptr->nzmax*2);
  }
  Ai[ptr->nnz] = i;
  Aj[ptr->nnz] = j;
  Av[ptr->nnz] = v;
  ++ptr->nnz;
}

cholmod_sparse *SparseMatrix::selfProduct() const
{
  auto A = this->to_sparse();
  auto At = cholmod_transpose(A, 2, global::cm);
  auto AtA = cholmod_aat(At, NULL, 0, 1, global::cm);

  cholmod_free_sparse(&A, global::cm);
  cholmod_free_sparse(&At, global::cm);
  return AtA;
}

cholmod_sparse *SparseMatrix::to_sparse(bool isSym) const
{
  auto A = cholmod_triplet_to_sparse(ptr, ptr->nzmax, global::cm);
  if(isSym) A->stype = 1;
  return A;
}

DenseVector SparseMatrix::solve(const DenseVector &b, bool isSym)
{
  cout << "solving A\b ..." << endl;
  auto A = this->to_sparse(isSym);

  cholmod_factor *L = cholmod_analyze(A, global::cm);
  cholmod_factorize(A, L, global::cm);
  DenseVector x(b.length());

  auto y = cholmod_solve(CHOLMOD_A, L, b.ptr, global::cm);
  memcpy(x.ptr->x, y->x, sizeof(double)*x.length());

  cout << "solved." << endl;
  cholmod_free_sparse(&A, global::cm);
  cholmod_free_factor(&L, global::cm);
  cholmod_free_dense(&y, global::cm);
  return x;
}

DenseVector SparseMatrix::operator*(const DenseVector &b)
{
  DenseVector x(b.length());
  double alpha[2] = {1, 0};
  double beta[2] = {0, 0};

  auto A = this->to_sparse();
  cholmod_sdmult(A, 0, alpha, beta, b.ptr, x.ptr, global::cm);

  cholmod_free_sparse(&A, global::cm);
  return x;
}

ostream &operator<<(ostream &os, const SparseMatrix &M)
{
  for(int i=0;i<M.ptr->nnz;++i) {
    os << M.Ai[i] << " " << M.Aj[i] << " " << M.Av[i] << endl;
  }
  return os;
}

void SparseMatrix::resize(int nzmax)
{
  auto newptr = cholmod_allocate_triplet(ptr->nrow, ptr->ncol, nzmax, ptr->stype, ptr->xtype, global::cm);
  if( newptr == nullptr ) {
    throw "unable to create sparse matrix. out of memory.";
  }
  // copy over the stuff in the old matrix
  memcpy(newptr->i, Ai, sizeof(int)*ptr->nnz);
  memcpy(newptr->j, Aj, sizeof(int)*ptr->nnz);
  memcpy(newptr->x, Av, sizeof(double)*ptr->nnz);
  newptr->nnz = ptr->nnz;

  cholmod_free_triplet(&ptr, global::cm);
  ptr = newptr;
  resetPointers();
}

void SparseMatrix::resetPointers()
{
  Ai = (int*)ptr->i;
  Aj = (int*)ptr->j;
  Av = (double*)ptr->x;
}
