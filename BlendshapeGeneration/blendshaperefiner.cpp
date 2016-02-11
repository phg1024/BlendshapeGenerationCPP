#include "blendshaperefiner.h"

void BlendshapeRefiner::LoadTemplateMeshes(const string &path, const string &basename) {
  A.resize(num_shapes + 1);
}

void BlendshapeRefiner::LoadInputReconstructionResults(const string &settings_filename) {

}

void BlendshapeRefiner::LoadInputPointClouds(const string &path) {

}

void BlendshapeRefiner::CreateTrainingShapes() {

}

void BlendshapeRefiner::InitializeBlendshapes() {

}

void BlendshapeRefiner::Refine() {
  // [Step 1]: deform the inintial training shapes with the input point clouds
  CreateTrainingShapes();

  // [Step 2]: create a set of initial blendshapes using initial neutral face mesh and template blendshapes
  InitializeBlendshapes();

  // [Step 3]: blendshape refinement
  {
    // [Step a]: Refine blendshapes

    // [Step b]: Update expression weights
  }
}