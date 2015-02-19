#pragma once

#include "common.h"
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

#include "cholmod.h"

template <typename T>
struct SparseMatrix {
  SparseMatrix(){}
  SparseMatrix(int m, int n) :rows(m), cols(n){}

  void reserve(int n) {
    Ai.reserve(n);
    Aj.reserve(n);
    Av.reserve(n);
  }

  void resize(int n) {
    Ai.resize(n);
    Aj.resize(n);
    Av.resize(n);
  }

  void write(const string &filename) {
    ofstream fout(filename);
    for (int i = 0; i < Ai.size(); ++i) {
      fout << Ai[i] << ' ' << Aj[i] << ' ' << Av[i] << endl;
    }
    fout.close();
  }

  void createCSRindices() {
    Ai_CSR.resize(Ai.back()+2, 0);
    for (int i = 0; i < Ai.size(); ++i) ++Ai_CSR[Ai[i]+1];
    // compute prefix sum
    for (int i = 1; i < Ai_CSR.size(); ++i) {
      Ai_CSR[i] += Ai_CSR[i - 1];
    }
  }

  vector<T> solve(vector<T> &rhs, cholmod_common *c);
  vector<T> operator*(const vector<T> &b);
  SparseMatrix<T> operator*(const SparseMatrix<T> &m);

  SparseMatrix<T> transposed() const;

  // computes A' * A
  SparseMatrix<T> selfProduct() const;

  cholmod_sparse* convertToCSC(cholmod_common *common) const;

  template <typename MT>
  friend ostream& operator<<(ostream &os, const SparseMatrix<MT> &m);

  int rows, cols;
  vector<int> Ai, Aj;
  vector<int> Ai_CSR;
  vector<T> Av;

  vector<int> Ai_CSC, Aj_CSC, Av_CSC;
};

template <typename T>
SparseMatrix<T> SparseMatrix<T>::selfProduct() const
{
  SparseMatrix<T> res(cols, cols);

  // transpose it first
  auto At = this->transposed();
  
  At.createCSRindices();

  // compute the product
  int n = Ai.size();

  res.reserve(n * 2);

  for(int x : At.Ai_CSR) cout << x << " ";
  cout << endl;

  for (int i = 0; i < cols; ++i) {
    int istart = At.Ai_CSR[i];
    int iend = At.Ai_CSR[i + 1];
    for (int j = 0; j < cols; ++j) {
      // compute the dot product of row i in A and column j in At
      // which is the same as row i in A and row j in At
      int jstart = At.Ai_CSR[j];
      int jend = At.Ai_CSR[j+1];

      T sum = 0;
      // scan through both rows and compute valid entries
      int k = istart, l = jstart;
      while (k < iend || l < jend) {
        // get the elements
        if (k < iend && l < jend) {
          int rA = At.Ai[k], cA = At.Aj[k];
          int rAt = At.Ai[l], cAt = At.Aj[l];

          if (cA < cAt) {
            // increment k
            ++k;
          }
          else if (cA > cAt) {
            // increment l
            ++l;
          }
          else if (cA == cAt) {
            // new entry, increment both k and l
            sum += At.Av[k] * At.Av[l];
            ++k; ++l;
          }
        }
        else {
          // no more entries, break it
          break;
        }
      }
      if (sum != 0) {
        // we have a valid entry, add it to res
        res.Ai.push_back(i); res.Aj.push_back(j); res.Av.push_back(sum);
      }
    }
  }

  return res;
}

template <typename T>
cholmod_sparse* SparseMatrix<T>::convertToCSC(cholmod_common *common) const
{
    cout << "converting to CSC format ..." << endl;
    int n = Av.size();

    // create a cholmod_triplet structure
    cholmod_triplet *t = cholmod_allocate_triplet(rows, cols, n, 0, CHOLMOD_REAL, common);

    // need to store this in column major
    memcpy(t->i, &(Ai[0]), sizeof(int)*n);
    memcpy(t->j, &(Aj[0]), sizeof(int)*n);
    double *Tx = (double*)(t->x);
    for(int i=0;i<n;++i) Tx[i] = Av[i];
    t->nnz = n;
    t->stype = 0;

    cholmod_sparse* s = cholmod_triplet_to_sparse(t, n, common);
    cholmod_free_triplet(&t, common);

    return s;
}

template <typename T>
ostream& operator<<(ostream &os, const SparseMatrix<T> &m) {
  int k = 0;
  int n = m.Ai.size();
  for (int i = 0; i < m.rows; ++i) {
    for (int j = 0; j < m.cols; ++j) {
      int idx = i*m.cols + j;
      if (k < n) {
        int midx = m.Ai[k] * m.cols + m.Aj[k];
        if (idx == midx) {
          os << m.Av[k];
          ++k;
        }
        else {
          os << 0;
        }
      }
      else {
        os << 0;
      }
      os << (j == m.cols - 1 ? "\n" : " ");
    }
  }
  return os;
}


template <typename T>
SparseMatrix<T> SparseMatrix<T>::transposed() const
{
  SparseMatrix<T> res(cols, rows);

  int n = Ai.size();
  vector<int> colCount(cols, 0);
  for (int i = 0; i < n; ++i) ++colCount[Aj[i]];
  res.resize(n);
  vector<int> colStart(cols, 0);
  // compute the correct column starting index
  for (int i = 1; i < cols;++i) {
    colStart[i] = colCount[i - 1] + colStart[i-1];
  }
  // fill in the elements
  for (int i = 0; i < n; ++i) {
    int r = Ai[i], c = Aj[i];
    T v = Av[i];
    int pos = colStart[c];
    res.Ai[pos] = c; res.Aj[pos] = r; res.Av[pos] = v;
    ++colStart[c];    
  }
  return res;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*(const SparseMatrix<T> &M)
{
  SparseMatrix<T> res(rows, M.cols);
  
  // transpose it first
  auto Mt = M.transposed();

  this->createCSRindices();
  Mt.createCSRindices();

  // compute the product
  int n = Ai.size();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      throw "Not finished yet.";
    }
  }

  return res;
}

template <typename T>
vector<T> SparseMatrix<T>::operator*(const vector<T> &rhs)
{
  vector<T> lhs(rhs.size(), 0);
  int n = Ai.size();
  for (int i = 0; i < n; ++i) {
    int r = Ai[i], c = Aj[i];
    T v = Av[i];
    lhs[r] += v * rhs[c];
  }
  return lhs;
}

template <typename T>
vector<T> SparseMatrix<T>::solve(vector<T> &b0, cholmod_common *c)
{
    cout << "solving A\\b" << endl;
    int n = b0.size();
    cholmod_dense *x, *b;

    for(int i=0;i<b0.size();++i) cout << b0[i] << ' ';
    cout << endl;

    cout << "A = \n" << (*this) << endl;

    b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
    memcpy(b->x, &(b0[0]), sizeof(T)*n);
    double *pb = (double*)(b->x);
    for(int i=0;i<n;++i) cout << pb[i] << ' ';
    cout << endl;

    cholmod_sparse *A = this->convertToCSC(c);
    A->stype = 1;           // it is actually symmetric
    cholmod_print_sparse(A, "A", c);
    FILE *file = fopen("A.mat", "w");
    cholmod_write_sparse(file, A, 0, 0, c);


    cholmod_factor *L = cholmod_analyze(A, c);
    cholmod_print_factor(L, "L", c);

    cholmod_factorize(A, L, c);
    cholmod_print_sparse(A, "A factored", c);

    x = cholmod_solve(CHOLMOD_A, L, b, c);
    cholmod_print_dense(x, "x", c);

    cout << "solved." << endl;

    vector<T> res(n);
    double *X = (double*)x->x;
    for(int i=0;i<n;++i) {
        cout << X[i] << ' ';
        res[i] = X[i];
    }
    cout << endl;
    cholmod_free_factor (&L, c) ;		    /* free matrices */
    cholmod_free_sparse (&A, c) ;
    cholmod_free_dense (&x, c) ;
    cholmod_free_dense (&b, c) ;

    return res;
}
