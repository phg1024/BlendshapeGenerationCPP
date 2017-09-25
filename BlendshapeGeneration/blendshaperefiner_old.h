//
// Created by phg on 9/24/17.
//

#ifndef BLENDSHAPEGENERATION_BLENDSHAPEREFINER_OLD_H
#define BLENDSHAPEGENERATION_BLENDSHAPEREFINER_OLD_H


#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/utils.hpp>
#include <MultilinearReconstruction/statsutils.h>

#include "meshdeformer.h"
#include "meshtransferer.h"
#include "ndarray.hpp"
#include "cereswrapper.h"
#include "triangle_gradient.h"

Array1D<double> estimateWeights(const BasicMesh &S,
                                const BasicMesh &B0,
                                const vector<MatrixX3d> &dB,
                                const Array1D<double> &w0,  // init value
                                const Array1D<double> &wp,  // prior
                                double w_prior,
                                int itmax);

vector<BasicMesh> refineBlendShapes(const vector<BasicMesh> &S,
                                    const vector<Array2D<double>> &Sgrad,
                                    const vector<BasicMesh> &B,
                                    const vector<Array1D<double>> &alpha,
                                    double beta, double gamma,
                                    const Array2D<double> prior,
                                    const Array2D<double> w_prior,
                                    const vector<int> stationary_indices);

pair<VectorXd, BasicMesh> estimateNeutralFace(const BasicMesh& target);

void blendShapeGeneration();

#endif //BLENDSHAPEGENERATION_BLENDSHAPEREFINER_OLD_H
