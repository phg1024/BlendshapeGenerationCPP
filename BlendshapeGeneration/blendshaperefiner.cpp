#include "blendshaperefiner.h"

#include "meshdeformer.h"
#include "meshtransferer.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;

BlendshapeRefiner::BlendshapeRefiner() {
  model.reset(new MultilinearModel("/home/phg/Data/Multilinear/blendshape_core.tensor"));
  model_prior.reset(new MultilinearModelPrior());
  model_prior->load("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                    "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  template_mesh.reset(new BasicMesh("/home/phg/Data/Multilinear/template.obj"));
}

void BlendshapeRefiner::LoadTemplateMeshes(const string &path, const string &basename) {
  A.resize(num_shapes + 1);

  for(int i=0;i<=num_shapes;++i) {
    A[i].LoadOBJMesh(path + "shape" + to_string(i) + ".obj");
  }
}

void BlendshapeRefiner::LoadInputReconstructionResults(const string &settings_filename) {
  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);

  fs::path settings_filepath(settings_filename);
  fs::path image_files_path = settings_filepath.parent_path();

  for(auto& p : image_points_filenames) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second);
    fs::path res_filename = settings_filepath.parent_path() / fs::path(p.first + ".res");
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string());
    auto recon_results = LoadReconstructionResult(res_filename.string());
    image_bundles.push_back(ImageBundle(image_points_pair.first, image_points_pair.second, recon_results));
  }

  // Set the number of poses
  num_poses = image_points_filenames.size();
}

MatrixXd BlendshapeRefiner::LoadPointCloud(const string &filename) {
  ifstream fin(filename);
  vector<Vector3d> points;
  points.reserve(100000);
  while(fin) {
    double x, y, z;
    fin >> x >> y >> z;
    points.push_back(Vector3d(x, y, z));
  }
  MatrixXd P(points.size(), 3);
  for(int i=0;i<points.size();++i) {
    P.row(i) = points[i];
  }
  return P;
}

void BlendshapeRefiner::LoadInputPointClouds(const string &path) {
  // Load all points files
  point_clouds.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    point_clouds[i] = LoadPointCloud(path + "point_cloud_opt_raw" + to_string(i) + ".txt");
  }
}

void BlendshapeRefiner::CreateTrainingShapes() {
  S0.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    ColorStream(ColorOutput::Green)<< "creating initial training shape " << i;
    model->ApplyWeights(image_bundles[i].params.params_model.Wid, image_bundles[i].params.params_model.Wexp);
    S0[i] = *template_mesh;
    S0[i].UpdateVertices(model->GetTM());
    S0[i].ComputeNormals();
    S0[i].Write("Sinit_" + to_string(i) + ".obj");
  }
  ColorStream(ColorOutput::Blue)<< "initial training shapes created.";

  MeshDeformer deformer;
  S.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    ColorStream(ColorOutput::Green)<< "creating refined training shape " << i;
    deformer.setSource(S0[i]);
    S[i] = deformer.deformWithPoints(point_clouds[i], PointCloud(), 20);
    S[i].Write("S_" + to_string(i) + ".obj");
  }
  ColorStream(ColorOutput::Blue)<< "refined training shapes created.";
}

void BlendshapeRefiner::InitializeBlendshapes() {
  // Create the initial neutral face mesh
  model->ApplyWeights(image_bundles[0].params.params_model.Wid, model_prior->Wexp0);

  Binit.resize(A.size());
  Binit[0] = *template_mesh;
  Binit[0].UpdateVertices(model->GetTM());
  Binit[0].ComputeNormals();

  // Deformation transfer to obtain all other blendshapes
  auto& B0 = Binit[0];
  vector<int> stationary_indices = B0.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  }) ;
  auto& A0 = A[0];
  MeshTransferer transferer;
  transferer.setSource(A0); // set source and compute deformation gradient
  transferer.setTarget(B0); // set target and compute deformation gradient
  transferer.setStationaryVertices(stationary_indices);

  for(int i=1;i<=num_shapes;++i) {
    ColorStream(ColorOutput::Green)<< "creating shape " << i;
    Binit[i] = transferer.transfer(A[i]);
    Binit[i].Write("Binit_" + to_string(i) + ".obj");
  }
  B = Binit;
  ColorStream(ColorOutput::Blue)<< "initial set of blendshapes created.";
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