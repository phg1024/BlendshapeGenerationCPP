#ifndef DENSEVECTOR_H
#define DENSEVECTOR_H

#include "common.h"

#include "cholmod.h"

class DenseVector
{
public:
    DenseVector();
    DenseVector(int n);
    ~DenseVector();

    int length() const { return ptr->nrow; }
    double& operator()(int i) { return x[i]; }
    const double& operator()(int i) const { return x[i]; }

private:
    friend class SparseMatrix;
    friend class DenseMatrix;

    double *x;
    cholmod_dense *ptr;
};

#endif // DENSEVECTOR_H
