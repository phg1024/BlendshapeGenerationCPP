#include "common.h"

#include "blendshapegeneration.h"
#include <QtWidgets/QApplication>

#include "testcases.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/utils.hpp>

#include "meshdeformer.h"
#include "meshtransferer.h"
#include "cereswrapper.h"

#include "Geometry/matrix.hpp"
#include "triangle_gradient.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>

void laplacianDeformation() {
  const string datapath = "/home/phg/Data/FaceWarehouse_Data_0/";

  BasicMesh m;
  m.LoadOBJMesh(datapath + "Tester_1/Blendshape/shape_0.obj");

  MeshDeformer deformer;
  deformer.setSource(m);

  vector<int> landmarks = LoadIndices(datapath+"landmarks_74_new.txt");
  deformer.setLandmarks(landmarks);

  int objidx = 1;
  BasicMesh T;
  T.LoadOBJMesh(datapath + "Tester_1/TrainingPose/pose_" + to_string(objidx) + ".obj");

  PointCloud lm_points;
  lm_points.points.resize(landmarks.size(), 3);
  for(int i=0;i<landmarks.size();++i) {
    auto vi = T.vertex(landmarks[i]);
    lm_points.points(i, 0) = vi[0];
    lm_points.points(i, 1) = vi[1];
    lm_points.points(i, 2) = vi[2];
  }
  BasicMesh D = deformer.deformWithMesh(T, lm_points, 20);

  D.Write("deformed" + to_string(objidx) + ".obj");
}

void laplacianDeformation_pointcloud() {
  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string res_filename("/home/phg/Data/InternetRecon/yaoming/4.jpg.res");
  const string pointcloud_filename("/home/phg/Data/InternetRecon/yaoming/SFS/point_cloud_opt_raw4.txt");

  // Update the mesh with the blendshape weights
  MultilinearModel model(model_filename);
  auto recon_results = LoadReconstructionResult(res_filename);
  model.ApplyWeights(recon_results.params_model.Wid, recon_results.params_model.Wexp);

  BasicMesh m;
  m.LoadOBJMesh(datapath + "Tester_1/Blendshape/shape_0.obj");
  m.UpdateVertices(model.GetTM());
  m.ComputeNormals();

  m.Write("source.obj");

  MeshDeformer deformer;
  deformer.setSource(m);

  vector<int> landmarks = LoadIndices(datapath+"landmarks_74_new.txt");
  //deformer.setLandmarks(landmarks);

  ifstream fin(pointcloud_filename);
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

  PointCloud lm_points;
  lm_points.points.resize(0, 0);

  // Filter the faces to reduce the search range
  vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
    Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
    return c[2] > -0.5;
  });
  deformer.setValidFaces(valid_faces);

  BasicMesh D = deformer.deformWithPoints(P, lm_points, 20);

  D.Write("deformed.obj");
}

void printUsage() {
  cout << "Laplacian deformation: [program] -l" << endl;
  cout << "Laplacian deformation with point cloud: [program] -lp" << endl;
}

int main(int argc, char *argv[])
{
  google::InitGoogleLogging(argv[0]);

#if RUN_TESTS
  TestCases::testCeres();
  return 0;
#else

  if( argc < 2 ) {
    printUsage();
    return 0;
  }

  string option = argv[1];

  if( option == "-l" ) {
    laplacianDeformation();
  }
  else if( option == "-lp" ) {
    laplacianDeformation_pointcloud();
  }

  return 0;
#endif
}
