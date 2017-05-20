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
  m.BuildHalfEdgeMesh();
  cout << "subdivide..." << endl;
  m.Subdivide();
  m.BuildHalfEdgeMesh();
  cout << "subdivide..." << endl;
  m.Subdivide();

  MeshDeformer deformer;
  deformer.setSource(m);

  vector<int> landmarks = LoadIndices(datapath+"landmarks_74_new.txt");
  deformer.setLandmarks(landmarks);

  vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
    Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
    return c[2] > -0.75;
  });
  deformer.setValidFaces(valid_faces);

  int objidx = 1;
  BasicMesh T;
  T.LoadOBJMesh(datapath + "Tester_1/TrainingPose/pose_" + to_string(objidx) + ".obj");
  T.BuildHalfEdgeMesh();
  T.Subdivide();
  T.BuildHalfEdgeMesh();
  T.Subdivide();

  MatrixX3d lm_points(landmarks.size(), 3);
  for(int i=0;i<landmarks.size();++i) {
    auto vi = T.vertex(landmarks[i]);
    lm_points(i, 0) = vi[0];
    lm_points(i, 1) = vi[1];
    lm_points(i, 2) = vi[2];
  }
  BasicMesh D = deformer.deformWithMesh(T, lm_points, 20);

  D.Write("deformed" + to_string(objidx) + ".obj");
}

void laplacianDeformation_pointcloud(
  const string& res_filename,
  const string& pointcloud_filename,
  const string& output_mesh_filename
) {
  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string mesh_filename(datapath + "Tester_1/Blendshape/shape_0.obj");
  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");

  // Update the mesh with the blendshape weights
  MultilinearModel model(model_filename);
  auto recon_results = LoadReconstructionResult(res_filename);
  cout << "Recon results loaded." << endl;
  model.ApplyWeights(recon_results.params_model.Wid, recon_results.params_model.Wexp);

  glm::dmat4 R = glm::eulerAngleZ(-recon_results.params_model.R[2])
               * glm::eulerAngleX(-recon_results.params_model.R[1])
               * glm::eulerAngleY(-recon_results.params_model.R[0]);

  ifstream fin(pointcloud_filename);
  vector<Vector3d> points;
  points.reserve(100000);
  while(fin) {
    double x, y, z;
    fin >> x >> y >> z;

    // rotate the input point cloud to regular view
    glm::dvec4 pt0 =  R * glm::dvec4(x, y, z, 1.0);

    points.push_back(Vector3d(pt0.x, pt0.y, pt0.z));
  }
  cout << "Points loaded: " << points.size() << " points." << endl;

  MatrixXd P(points.size(), 3);
  for(int i=0;i<points.size();++i) {
    P.row(i) = points[i];
  }

  MatrixX3d lm_points;

  BasicMesh m;
  m.LoadOBJMesh(mesh_filename);
  m.UpdateVertices(model.GetTM());

  const int max_iters = 1;
  for(int i=0;i<max_iters;++i) {
    m.Write("source" + to_string(i) + ".obj");

    MeshDeformer deformer;
    deformer.setSource(m);

    // Filter the faces to reduce the search range
    vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
      Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
      return c[2] > -0.5;
    });
    deformer.setValidFaces(valid_faces);
    // No landmarks
    deformer.setLandmarks(vector<int>());

    BasicMesh D = deformer.deformWithPoints(P, lm_points, 20);
    D.Write("deformed" + to_string(i) + ".obj");

    // replace it
    m = D;

    if(max_iters > 1) {
      m.BuildHalfEdgeMesh();
      m.Subdivide();
    }
  }

  m.Write(output_mesh_filename);
}

void printUsage() {
  cout << "Laplacian deformation: [program] -l" << endl;
  cout << "Laplacian deformation with point cloud: [program] -lp" << endl;
}

int main(int argc, char *argv[])
{
  google::InitGoogleLogging(argv[0]);
  QApplication a(argc, argv);
  glutInit(&argc, argv);

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
    string res_file = argc>2?argv[2]:"/home/phg/Data/InternetRecon0/yaoming/4.jpg.res";
    string pointcloud_file = argc>3?argv[3]:"/home/phg/Data/InternetRecon0/yaoming/SFS/masked_optimized_point_cloud_7.txt";
    string output_mesh_file = argc>4?argv[4]:"deformed.obj";
    laplacianDeformation_pointcloud(res_file, pointcloud_file, output_mesh_file);
  }

  return 0;
#endif
}
