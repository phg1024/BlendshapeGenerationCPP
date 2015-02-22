#pragma once

#include "common.h"

#include "cholmod.h"
#include "densevector.h"

class SparseMatrix {
public:
    SparseMatrix(){}
    SparseMatrix(int m, int n, int nzmax = 64);
    SparseMatrix(const SparseMatrix &other);
    SparseMatrix &operator=(const SparseMatrix&);

    void append(int i, int j, double v);

    cholmod_sparse *selfProduct() const;

    cholmod_sparse *to_sparse(bool isSym = false) const;
    DenseVector solve(const DenseVector &b, bool isSym = false);
    DenseVector operator*(const DenseVector &b);

    friend ostream& operator<<(ostream& os, const SparseMatrix &M);
protected:
    void resize(int nzmax);
    void resetPointers();

private:
    cholmod_triplet *ptr;
    int *Ai, *Aj;
    double *Av;
};
