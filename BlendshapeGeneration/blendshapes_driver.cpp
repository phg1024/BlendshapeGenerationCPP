#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <QApplication>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include "MultilinearReconstruction/common.h"
#include "MultilinearReconstruction/basicmesh.h"
#include "MultilinearReconstruction/constraints.h"
#include "MultilinearReconstruction/costfunctions.h"
#include "MultilinearReconstruction/ioutilities.h"
#include "MultilinearReconstruction/multilinearmodel.h"
#include "MultilinearReconstruction/parameters.h"
#include "MultilinearReconstruction/statsutils.h"
#include "MultilinearReconstruction/utils.hpp"

#include "MultilinearReconstruction/OffscreenMeshVisualizer.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;

using namespace Eigen;

vector<BasicMesh> LoadBlendshapes(const string& path) {
  const string prefix = "B_";
  const int num_blendshapes = 47;
  vector<BasicMesh> blendshapes(num_blendshapes);
  #pragma omp parallel for
  for(int i=0;i<num_blendshapes;++i) {
    blendshapes[i].LoadOBJMesh(path + "/" + prefix + to_string(i) + ".obj");
    blendshapes[i].ComputeNormals();
  }

  return blendshapes;
}

void VisualizeReconstructionResult(
  const QImage& img,
  const string& res_filename,
  const vector<BasicMesh>& blendshapes,
  const string& rendering_settings_filename,
  const string& output_image_filename,
  bool use_head_pose,
  bool scale_output=true) {

  int imgw = img.width();
  int imgh = img.height();
  if(scale_output) {
    const int target_size = 640;
    double scale = static_cast<double>(target_size) / imgw;
    imgw *= scale;
    imgh *= scale;
  }

  BasicMesh mesh = blendshapes.front();
  auto recon_results = LoadReconstructionResult(res_filename);

  // Interpolate the blendshapes
  {
    MatrixX3d v = mesh.vertices();
    for(int i=1;i<47;++i) {
      const double weight_i = recon_results.params_model.Wexp_FACS[i];
        if(weight_i > 0) v += (blendshapes[i].vertices() - mesh.vertices()) * weight_i;
    }
    mesh.vertices() = v;
  }
  mesh.ComputeNormals();

  OffscreenMeshVisualizer visualizer(imgw, imgh);

  visualizer.LoadRenderingSettings(rendering_settings_filename);
  visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
  visualizer.SetRenderMode(OffscreenMeshVisualizer::TexturedMesh);
  visualizer.BindMesh(mesh);
  visualizer.BindTexture(img);

  visualizer.SetCameraParameters(recon_results.params_cam);
  if(use_head_pose)
    visualizer.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
  else
    visualizer.SetMeshRotationTranslation(Vector3d(0, 0, 0), Vector3d(0, 0, -12));
  visualizer.SetIndexEncoded(false);
  visualizer.SetEnableLighting(true);

  QImage output_img = visualizer.Render(true);
  output_img.save(output_image_filename.c_str());
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  if(argc<9) {
    cout << "Usage: " << argv[0] << " texture_img res_path prefix start_idx end_idx blendshapes_path output_path rendering_settings" << endl;
    return 1;
  }

  const string texture_filename = argv[1];
  const string res_path = argv[2];
  const string prefix = argv[3];
  const int start_idx = std::stoi(string(argv[4]));
  const int end_idx = std::stoi(string(argv[5]));
  const string blendshapes_path = argv[6];
  const string output_path = argv[7];
  const string rendering_settings_filename = argv[8];

  bool use_head_pose = true;
  if(argc > 9 && string(argv[9]) == "no_pose") use_head_pose = false;

  vector<BasicMesh> blendshapes = LoadBlendshapes(blendshapes_path);
  QImage texture(texture_filename.c_str());

  #pragma omp parallel for
  for(int idx=start_idx;idx<=end_idx;++idx) {
    char buff[100];
    snprintf(buff, sizeof(buff), "%05d", idx);
    string idx_str = buff;
    string res_filename = res_path + "/" + prefix + idx_str + ".jpg.res";
    string output_filename = output_path + "/" + idx_str + ".png";
    cout << "Rendering to " << output_filename << endl;
    VisualizeReconstructionResult(texture, res_filename, blendshapes,
      rendering_settings_filename, output_filename, use_head_pose);
  }
  return 0;
}
