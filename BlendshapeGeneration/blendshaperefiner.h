#ifndef FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
#define FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H

#include "common.h"

#include "ndarray.hpp"
#include "triangle_gradient.h"
#include "pointcloud.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/reporter.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;

#include "json/src/json.hpp"
using json = nlohmann::json;

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
  BlendshapeRefiner(json settings = json{});
  ~BlendshapeRefiner() {}

  void SetBlendshapeCount(int count) { num_shapes = count; }
  void SetResourcesPath(const string& path);
  void SetReconstructionsPath(const string& path);
  void SetPointCloudsPath(const string& path);
  void SetInputBlendshapesPath(const string& path);
  void SetBlendshapesPath(const string& path);

  void LoadTemplateMeshes(const string& path, const string& basename);
  void LoadSelectionFile(const string& selection_filename);
  void LoadInputReconstructionResults(const string& settings_filename);
  void LoadInputPointClouds();

  void Refine();
  void Refine_EBFR();

private:
  void LoadInitialBlendshapes();
  void CreateTrainingShapes();
  void InitializeBlendshapes();

  MatrixXd LoadPointCloud(const string& filename, const glm::dmat4& R);

  vector <BasicMesh> RefineBlendshapes(const vector <BasicMesh> &S,
                                       const vector <vector<PhGUtils::Matrix3x3d>> &Sgrad,
                                       const vector <BasicMesh> &A,
                                       const vector <BasicMesh> &B, const BasicMesh &B00,
                                       const vector <VectorXd> &alpha,
                                       double beta, double gamma,
                                       const vector <vector<PhGUtils::Matrix3x3d>> &prior,
                                       const MatrixXd& w_prior,
                                       const vector<int> &stationary_vertices,
                                       const vector<bool> &stationary_faces_set,
                                       bool refine_neutral_only = true);

  vector <BasicMesh> RefineBlendshapes_EBFR(const vector <BasicMesh> &S,
                                            const vector <vector<PhGUtils::Matrix3x3d>> &Sgrad,
                                            const vector <BasicMesh> &A,
                                            const vector <BasicMesh> &B, const BasicMesh &B00,
                                            const vector <VectorXd> &alpha,
                                            double beta, double gamma,
                                            const vector <vector<PhGUtils::Matrix3x3d>> &prior,
                                            const MatrixXd& w_prior,
                                            const vector<int> stationary_indices);


  VectorXd EstimateWeights(const BasicMesh &S,
                           const BasicMesh &B0,
                           const vector<MatrixX3d> &dB,
                           const VectorXd &w0,  // init value
                           const VectorXd &wp,  // prior
                           double w_prior,
                           int itmax,
                           const vector<int>& valid_vertices = vector<int>());

protected:
  string FullFile(const fs::path& path, const string& filename) {
    return (path / fs::path(filename)).string();
  }
  string InBlendshapesDirectory(const string& filename) {
    return FullFile(blendshapes_path, filename);
  }

private:
  MultilinearModel model;
  MultilinearModelPrior model_prior;
  BasicMesh template_mesh;

  int num_shapes;
  vector<BasicMesh> A;      // template blendshapes

  vector<BasicMesh> Binit;  // transferred/initial blendshapes
  vector<BasicMesh> B;      // refined blendshapes

  int num_poses;
  bool use_init_blendshapes;
  bool do_subdivision;
  bool blendshapes_subdivided;
  vector<ImageBundle> image_bundles;
  vector<MatrixXd> point_clouds;
  vector<BasicMesh> S0;     // initial training shapes
  vector<BasicMesh> S;      // point cloud deformed training shapes
  vector<MatrixX3d> Slandmarks;

  fs::path resources_path, reconstructions_path, point_clouds_path, input_blendshapes_path;
  fs::path blendshapes_path;
  vector<int> selection_indices;
  map<int, int> selection_to_order_map;

  Reporter reporter;
};

#endif //FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
