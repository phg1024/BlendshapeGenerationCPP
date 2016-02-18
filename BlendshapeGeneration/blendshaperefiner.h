#ifndef FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
#define FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H

#include "common.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>

struct ImageBundle {
  ImageBundle() {}
  ImageBundle(const QImage& image, const vector<Constraint2D>& points, const ReconstructionResult& params)
    : image(image), points(points), params(params) {}
  QImage image;
  vector<Constraint2D> points;
  ReconstructionResult params;
};

class BlendshapeRefiner {
public:
  BlendshapeRefiner();
  ~BlendshapeRefiner() {}

  void SetBlendshapeCount(int count) { num_shapes = count; }
  void LoadTemplateMeshes(const string& path, const string& basename);
  void LoadInputReconstructionResults(const string& settings_filename);
  void LoadInputPointClouds(const string& path);

  void Refine();

private:
  void CreateTrainingShapes();
  void InitializeBlendshapes();

  MatrixXd LoadPointCloud(const string& filename);

private:
  unique_ptr<MultilinearModel> model;
  unique_ptr<MultilinearModelPrior> model_prior;
  unique_ptr<BasicMesh> template_mesh;

  int num_shapes;
  vector<BasicMesh> A;      // template blendshapes

  vector<BasicMesh> Binit;  // transferred blendshapes
  vector<BasicMesh> B;      // refined blendshapes

  int num_poses;
  vector<ImageBundle> image_bundles;
  vector<MatrixXd> point_clouds;
  vector<BasicMesh> S0;     // initial training shapes
  vector<BasicMesh> S;      // point cloud deformed training shapes
};

#endif //FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
