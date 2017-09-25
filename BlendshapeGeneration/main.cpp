#include "common.h"

#include "blendshapegeneration.h"
#include "interactiverigging.h"
#include <QtWidgets/QApplication>

#include "testcases.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/utils.hpp>
#include <MultilinearReconstruction/statsutils.h>

#include "blendshaperefiner.h"
#include "blendshaperefiner_old.h"
#include "meshdeformer.h"
#include "meshtransferer.h"
#include "cereswrapper.h"

#include "Geometry/matrix.hpp"
#include "triangle_gradient.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/program_options.hpp"
#include "boost/timer/timer.hpp"

namespace fs = boost::filesystem;

#include "json/src/json.hpp"
using json = nlohmann::json;

void blendShapeGeneration_mesh(
  const string& source_path,
  bool subdivision,
  bool disable_neutral_opt,
  bool mask_nose_and_fore_head
) {
  BlendshapeRefiner refiner(
    json{
    {"use_init_blendshapes", false},
    {"subdivision", subdivision},
    {"blendshapes_subdivided", false},
    {"mask_nose_and_fore_head", mask_nose_and_fore_head}
  });

  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(source_path);
  refiner.SetExampleMeshesPath(source_path + "/normal_constraints");
  refiner.SetInputBlendshapesPath("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/");
  refiner.SetBlendshapesPath(source_path + "/blendshapes");

  refiner.LoadSelectionFile("selection_sfs.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputExampleMeshes();

  refiner.Refine(false, disable_neutral_opt);
}

void blendShapeGeneration_mesh_blendshapes(
  const string& source_path,
  const string& recon_path,
  const string& meshes_path,
  const string& input_blendshapes_path,
  const string& blendshapes_path,
  bool subdivision,
  bool blendshapes_subdivided,
  bool initialize_only,
  bool disable_neutral_opt,
  bool mask_nose_and_fore_head
) {
  BlendshapeRefiner refiner(
    json{
    {"use_init_blendshapes", true},
    {"subdivision", subdivision},
    {"blendshapes_subdivided", blendshapes_subdivided},
    {"mask_nose_and_fore_head", mask_nose_and_fore_head}
  }
  );
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(recon_path);
  refiner.SetExampleMeshesPath(meshes_path);
  refiner.SetInputBlendshapesPath(input_blendshapes_path);
  refiner.SetBlendshapesPath(blendshapes_path);

  refiner.LoadSelectionFile("selection_sfs.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputExampleMeshes();

  refiner.Refine(initialize_only, disable_neutral_opt);
}

void blendShapeGeneration_pointcloud(
  const string& source_path,
  bool subdivision,
  bool disable_neutral_opt,
  bool mask_nose_and_fore_head) {
  BlendshapeRefiner refiner(
    json{
        {"use_init_blendshapes", false},
        {"subdivision", subdivision},
        {"blendshapes_subdivided", false},
        {"mask_nose_and_fore_head", mask_nose_and_fore_head}
      });
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(source_path);
  refiner.SetPointCloudsPath(source_path + "/SFS");
  refiner.SetInputBlendshapesPath("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/");
  refiner.SetBlendshapesPath(source_path + "/blendshapes");

  refiner.LoadSelectionFile("selection_sfs.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputPointClouds();

  refiner.Refine(false, disable_neutral_opt);
}

void blendShapeGeneration_pointcloud_blendshapes(
  const string& source_path,
  const string& recon_path,
  const string& point_clouds_path,
  const string& input_blendshapes_path,
  const string& blendshapes_path,
  bool subdivision,
  bool blendshapes_subdivided,
  bool initialize_only,
  bool disable_neutral_opt,
  bool mask_nose_and_fore_head
) {
  BlendshapeRefiner refiner(
    json{
      {"use_init_blendshapes", true},
      {"subdivision", subdivision},
      {"blendshapes_subdivided", blendshapes_subdivided},
      {"mask_nose_and_fore_head", mask_nose_and_fore_head}
    }
  );
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(recon_path);
  refiner.SetPointCloudsPath(point_clouds_path);
  refiner.SetInputBlendshapesPath(input_blendshapes_path);
  refiner.SetBlendshapesPath(blendshapes_path);

  refiner.LoadSelectionFile("selection_sfs.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputPointClouds();

  refiner.Refine(initialize_only, disable_neutral_opt);
}

void blendShapeGeneration_pointcloud_EBFR() {
  BlendshapeRefiner refiner;
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Storage/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  // yaoming
  refiner.SetResourcesPath("/home/phg/Storage/Data/InternetRecon0/yaoming/crop/");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputPointClouds();

  // // Turing
  // refiner.SetResourcesPath("/home/phg/Storage/Data/InternetRecon2/Allen_Turing/");
  // refiner.LoadInputReconstructionResults("setting.txt");
  // refiner.LoadInputPointClouds();

  refiner.Refine_EBFR();
}

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  google::InitGoogleLogging(argv[0]);
  glutInit(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help messages")
    ("oldfasion", "Generate blendshapes the old way.")
    ("pointclouds", "Generate blendshapes from point clouds")
    ("pointclouds_with_init_shapes", "Generate blendshapes from point clouds and a set of initial blendshapes")
    ("pointclouds_from_meshes", "Generate blendshapes from point clouds sampled from meshes")
    ("pointclouds_from_meshes_with_init_shapes", "Generate blendshapes from point clouds sampled from meshes and a set of initial blendshapes")
    ("ebfr", "Generate blendshapes using example based facial rigging method")
    ("rigging", "Interactive rigging mode with specified blendshapes")
    ("repo_path", po::value<string>(), "Path to images repo.")
    ("recon_path", po::value<string>(), "Path to large scale reconstruction results")
    ("pointclouds_path", po::value<string>(), "Path to input point clouds")
    ("meshes_path", po::value<string>(), "Path to input example meshes")
    ("init_blendshapes_path", po::value<string>(), "Path to initial blendshapes")
    ("blendshapes_path", po::value<string>(), "Path to output blendshapes")
    ("initialize_only", "Only perform initialization")
    ("no_neutral", "Disable neutral shape optimzation")
    ("subdivided", "Indicate the input blendshapes are subdivided")
    ("subdivision", "Enable subdivision")
    ("mask_nose_and_fore_head", "Use fore head and nose mask")
    ("ref_mesh", po::value<string>(), "Reference mesh for distance computation")
    ("mesh", po::value<string>(), "Mesh to visualize")
    ("vis", "Visualize blendshape mesh")
    ("rendering_settings", po::value<string>(), "Rendering settings to use")
    ("skip_faces", po::value<string>(), "Faces to skip rendering")
    ("texture", po::value<string>(), "Texture for the mesh")
    ("sideview", "Visualize the blendshape mesh in side view")
    ("silent", "Silent visualization using offscreen drawing")
    ("save,s", po::value<string>(), "Save the result to a file");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << endl;
      return 1;
    }

    if (vm.count("oldfasion")) {
      blendShapeGeneration();
    } else if (vm.count("ebfr")) {
      throw runtime_error("This is no longer supported.");
      blendShapeGeneration_pointcloud_EBFR();
    } else if (vm.count("rigging")) {
      QApplication a(argc, argv);
      InteractiveRigging rigging;
      rigging.LoadBlendshapes(vm["blendshapes_path"].as<string>());
      rigging.show();
      return a.exec();
    } else if (vm.count("pointclouds")) {
      if (vm.count("repo_path")) {
        blendShapeGeneration_pointcloud(vm["repo_path"].as<string>(),
                                        vm.count("subdivision"),
                                        vm.count("no_neutral"),
                                        vm.count("mask_nose_and_fore_head"));
      } else {
        throw po::error("Need to specify repo_path");
      }
    } else if (vm.count("pointclouds_from_meshes")) {
      if (vm.count("repo_path")) {
        blendShapeGeneration_mesh(vm["repo_path"].as<string>(),
                                  vm.count("subdivision"),
                                  vm.count("no_neutral"),
                                  vm.count("mask_nose_and_fore_head"));
      } else {
        throw po::error("Need to specify repo_path");
      }
    } else if (vm.count("pointclouds_with_init_shapes")) {
      if(vm.count("repo_path")
      && vm.count("recon_path")
      && vm.count("pointclouds_path")
      && vm.count("init_blendshapes_path")
      && vm.count("blendshapes_path")) {
        blendShapeGeneration_pointcloud_blendshapes(
          vm["repo_path"].as<string>(),
          vm["recon_path"].as<string>(),
          vm["pointclouds_path"].as<string>(),
          vm["init_blendshapes_path"].as<string>(),
          vm["blendshapes_path"].as<string>(),
          vm.count("subdivision"),
          vm.count("subdivided"),
          vm.count("initialize_only"),
          vm.count("no_neutral"),
          vm.count("mask_nose_and_fore_head")
        );
      } else {
        throw po::error("Need to specify repo_path, recon_path, pointclouds_path, init_blendshapes_path, blendshapes_path");
      }
    } else if (vm.count("pointclouds_from_meshes_with_init_shapes")) {
      if(vm.count("repo_path")
         && vm.count("recon_path")
         && vm.count("meshes_path")
         && vm.count("init_blendshapes_path")
         && vm.count("blendshapes_path")) {
        blendShapeGeneration_mesh_blendshapes(
          vm["repo_path"].as<string>(),
          vm["recon_path"].as<string>(),
          vm["meshes_path"].as<string>(),
          vm["init_blendshapes_path"].as<string>(),
          vm["blendshapes_path"].as<string>(),
          vm.count("subdivision"),
          vm.count("subdivided"),
          vm.count("initialize_only"),
          vm.count("no_neutral"),
          vm.count("mask_nose_and_fore_head")
        );
      } else {
        throw po::error("Need to specify repo_path, recon_path, pointclouds_path, init_blendshapes_path, blendshapes_path");
      }
    } else if(vm.count("vis")) {
      bool save_result = vm.count("save");
      bool compare_mode = vm.count("ref_mesh");
      bool sideview = vm.count("sideview");

      string input_mesh_file;
      if(vm.count("mesh")) {
        input_mesh_file = vm["mesh"].as<string>();
      } else {
        throw po::error("Need to specify mesh");
      }

      BlendshapeGeneration w(vm.count("silent"));
      if(!vm.count("silent")) w.show();
      w.SetSideView(sideview);

      if(vm.count("texture")) w.SetTexture(vm["texture"].as<string>());
      if(vm.count("rendering_settings")) w.LoadRenderingSettings(vm["rendering_settings"].as<string>());
      if(vm.count("skip_faces")) w.LoadSkipFaces(vm["skip_faces"].as<string>(), vm.count("subdivided"));

      if(compare_mode) {
        string ref_mesh_file = vm["ref_mesh"].as<string>();

        w.LoadMeshes(input_mesh_file, ref_mesh_file);
        w.setWindowTitle(input_mesh_file.c_str());
      } else {
        w.LoadMesh(input_mesh_file);
        w.setWindowTitle(input_mesh_file.c_str());
      }

      if(save_result) {
        cout << "Saving results ..." << endl;
        w.repaint();
        for(int i=0;i<10;++i)
          qApp->processEvents();

        w.Save(vm["save"].as<string>());
        qApp->processEvents();

        if(compare_mode) w.SaveError(vm["save"].as<string>() + ".error");
        return 0;
      } else {
        return a.exec();
      }
    }

  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    return 1;
  }

  return 0;
}
