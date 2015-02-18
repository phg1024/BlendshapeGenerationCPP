#include "blendshapegeneration.h"
#include <QtWidgets/QApplication>

#include "BasicMesh.h"
#include "MeshDeformer.h"

vector<int> loadLandmarks(const string &filename) {
  const int npts = 73;
  vector<int> v(npts);
  ifstream fin(filename);
  for (int i = 0; i < npts; ++i) fin >> v[i];
  return v;
}

void testMatrix() {
  BasicMatrix<float> A = BasicMatrix<float>::random(5, 5);
  cout << A << endl;
  auto B = A.inv();
  cout << B << endl;
  cout << B.transposed() << endl;
  cout << A * B << endl;
  cout << B * A << endl;
}

void testSparseMatrix() {
  /*
  1 0 0 0 1
  0 1 0 1 0
  0 0 1 0 0
  0 0 0 1 0
  */
  SparseMatrix<float> M(4, 5);
  M.resize(6);
  M.Ai = {0, 0, 1, 1, 2, 3};
  M.Aj = {0, 4, 1, 3, 2, 3};
  M.Av = {1, 2, 3, 4, 5, 6};

  cout << M << endl;
  cout << M.transposed() << endl;

  vector<float> b = { 1, 1, 1, 1, 1 };
  cout << M.selfProduct() << endl;
  
  auto x = M.solve(b);
  for (int i = 0; i < x.size(); ++i) cout << x[i] << ' ';
  cout << endl;
  vector<float> b1 = M * x;
  for (int i = 0; i < b1.size(); ++i) cout << b1[i] << ' ';
  cout << endl;
}

int main(int argc, char *argv[])
{
#if 0
  QApplication a(argc, argv);
  BlendshapeGeneration w;
  w.show();
  return a.exec();
#else
  testSparseMatrix();
  return 0;

  BasicMesh m;
  m.load("C:\\Users\\Peihong\\Desktop\\Data\\FaceWarehouse_Data_0\\Tester_1\\Blendshape\\shape_0.obj");

  MeshDeformer deformer;
  deformer.setSource(m);

  vector<int> landmarks = loadLandmarks("C:\\Users\\Peihong\\Desktop\\Code\\BlendshapeGeneration\\BlendshapeGeneration\\landmarks_74_new.txt");
  deformer.setLandmarks(landmarks);

  BasicMesh T;
  T.load("C:\\Users\\Peihong\\Desktop\\Data\\FaceWarehouse_Data_0\\Tester_1\\TrainingPose\\pose_1.obj");

  PointCloud lm_points;
  lm_points.points = T.verts.row(landmarks);
  BasicMesh D = deformer.deformWithMesh(T, lm_points);

  D.write("deformed.obj");
  return 0;
#endif
}
