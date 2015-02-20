#include "densevector.h"

DenseVector::DenseVector():ptr(nullptr){}

DenseVector::DenseVector(int n) {
    ptr = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, global::cm);
    x = (double*) ptr->x;
}

DenseVector::~DenseVector()
{
    cholmod_free_dense(&ptr, global::cm);
}

