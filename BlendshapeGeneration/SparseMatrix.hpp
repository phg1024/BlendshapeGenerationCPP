#pragma once

#include "common.h"
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

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
    for (int i = 0; i < Ai_CSR.size(); ++i) ++Ai_CSR[Ai[i]+1];
    // compute prefix sum
    for (int i = 1; i < Ai_CSR.size(); ++i) {
      Ai_CSR[i] += Ai_CSR[i - 1];
    }
  }

  vector<T> solve(vector<T> &rhs);  
  vector<T> solve_cg(const vector<T> &rhs);
  vector<T> operator*(const vector<T> &b);
  SparseMatrix<T> operator*(const SparseMatrix<T> &m);

  SparseMatrix<T> transposed() const;

  // computes A' * A
  SparseMatrix<T> selfProduct() const;

  template <typename MT>
  friend ostream& operator<<(ostream &os, const SparseMatrix<MT> &m);

  int rows, cols;
  vector<int> Ai, Aj;
  vector<int> Ai_CSR;
  vector<T> Av;
};

template <typename T>
vector<T> SparseMatrix<T>::solve_cg(const vector<T> &rhs)
{
  // can use this one if the original problem is an optimization problem
}

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
vector<T> SparseMatrix<T>::solve(vector<T> &b)
{
#if 1
  cout << "solving linear equation ..." << endl;
  int n = b.size();
  vector<T> y(n);
  int ldb = 1;  
  float alpha=1.0, beta = 0.0;
  char		matdescra[7] = "tlnc ";
  char transa = 't';
  vector<T> temp(n);
  mkl_scoosm(&transa, &rows, &cols, &alpha, matdescra, &(Av[0]), &(Ai[0]), &(Aj[0]), &n, &(b[0]), &ldb, &(temp[0]), &n);
  return y;
#else
  this->createCSRindices();

  MKL_INT mtype = 11;       /* Real unsymmetric matrix */
  /* RHS and solution vectors. */
  vector<float> x(b.size()), bs(b.size());
  double res, res0;
  MKL_INT nrhs = 1;     /* Number of right hand sides. */
  /* Internal solver memory pointer pt, */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
  /* or void *pt[64] should be OK on both architectures */
  void *pt[64];
  /* Pardiso control parameters. */
  MKL_INT iparm[64];
  MKL_INT maxfct, mnum, phase, error, msglvl;
  /* Auxiliary variables. */
  MKL_INT i, j;
  MKL_INT n = b.size();
  double ddum;          /* Double dummy */
  MKL_INT idum;         /* Integer dummy. */
  char *uplo;
  /* -------------------------------------------------------------------- */
  /* .. Setup Pardiso control parameters. */
  /* -------------------------------------------------------------------- */
  for (i = 0; i < 64; i++)
  {
    iparm[i] = 0;
  }
  iparm[0] = 1;         /* No solver default */
  iparm[1] = 2;         /* Fill-in reordering from METIS */
  iparm[3] = 0;         /* No iterative-direct algorithm */
  iparm[4] = 0;         /* No user fill-in reducing permutation */
  iparm[5] = 0;         /* Write solution into x */
  iparm[6] = 0;         /* Not in use */
  iparm[7] = 2;         /* Max numbers of iterative refinement steps */
  iparm[8] = 0;         /* Not in use */
  iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
  iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
  iparm[11] = 0;        /* Conjugate transposed/transpose solve */
  iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
  iparm[13] = 0;        /* Output: Number of perturbed pivots */
  iparm[14] = 0;        /* Not in use */
  iparm[15] = 0;        /* Not in use */
  iparm[16] = 0;        /* Not in use */
  iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1;       /* Output: Mflops for LU factorization */
  iparm[19] = 0;        /* Output: Numbers of CG Iterations */
  maxfct = 1;           /* Maximum number of numerical factorizations. */
  mnum = 1;         /* Which factorization to use. */
  msglvl = 1;           /* Print statistical information in file */
  error = 0;            /* Initialize error flag */
  /* -------------------------------------------------------------------- */
  /* .. Initialize the internal solver memory pointer. This is only */
  /* necessary for the FIRST call of the PARDISO solver. */
  /* -------------------------------------------------------------------- */
  for (i = 0; i < 64; i++)
  {
    pt[i] = 0;
  }
  /* -------------------------------------------------------------------- */
  /* .. Reordering and Symbolic Factorization. This step also allocates */
  /* all memory that is necessary for the factorization. */
  /* -------------------------------------------------------------------- */
  phase = 11;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
    &n, &(Av[0]), &(Ai_CSR[0]), &(Aj[0]), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  if (error != 0)
  {
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);
  }
  printf("\nReordering completed ... ");
  printf("\nNumber of nonzeros in factors = %d", iparm[17]);
  printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
  /* -------------------------------------------------------------------- */
  /* .. Numerical factorization. */
  /* -------------------------------------------------------------------- */
  phase = 22;
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
    &n, &(Av[0]), &(Ai_CSR[0]), &(Aj[0]), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
  if (error != 0)
  {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }
  printf("\nFactorization completed ... ");
  /* -------------------------------------------------------------------- */
  /* .. Back substitution and iterative refinement. */
  /* -------------------------------------------------------------------- */
  phase = 33;
  //  Loop over 3 solving steps: Ax=b, AHx=b and ATx=b
  for (i = 0; i < 3; i++)
  {
    iparm[11] = i;        /* Conjugate transposed/transpose solve */
    if (i == 0)
      uplo = "non-transposed";
    else if (i == 1)
      uplo = "conjugate transposed";
    else
      uplo = "transposed";

    printf("\n\nSolving %s system...\n", uplo);
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
      &n, &(Av[0]), &(Ai_CSR[0]), &(Aj[0]), &idum, &nrhs, iparm, &msglvl, &(b[0]), &(x[0]), &error);
    if (error != 0)
    {
      printf("\nERROR during solution: %d", error);
      exit(3);
    }

    printf("\nThe solution of the system is: ");
    for (j = 0; j < n; j++)
    {
      printf("\n x [%d] = % f", j, x[j]);
    }
    printf("\n");
    // Compute residual
    mkl_scsrgemv(uplo, &n, &(Av[0]), &(Ai_CSR[0]), &(Aj[0]), &(x[0]), &(bs[0]));
    res = 0.0;
    res0 = 0.0;
    for (j = 1; j <= n; j++)
    {
      res += (bs[j - 1] - b[j - 1]) * (bs[j - 1] - b[j - 1]);
      res0 += b[j - 1] * b[j - 1];
    }
    res = sqrt(res) / sqrt(res0);
    printf("\nRelative residual = %e", res);
    // Check residual
    if (res > 1e-10)
    {
      printf("Error: residual is too high!\n");
      exit(10 + i);
    }
  }

  /* -------------------------------------------------------------------- */
  /* .. Termination and release of memory. */
  /* -------------------------------------------------------------------- */
  phase = -1;           /* Release internal memory. */
  PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
    &n, &ddum, &(Ai_CSR[0]), &(Aj[0]), &idum, &nrhs,
    iparm, &msglvl, &ddum, &ddum, &error);

  return x;
#endif
}
