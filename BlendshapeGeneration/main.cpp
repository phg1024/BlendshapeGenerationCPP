#include "common.h"

#include "blendshapegeneration.h"
#include <QtWidgets/QApplication>

#include "densematrix.h"
#include "densevector.h"
#include "BasicMesh.h"
#include "MeshDeformer.h"

vector<int> loadLandmarks(const string &filename) {
  const int npts = 73;
  vector<int> v(npts);
  ifstream fin(filename);
  for (int i = 0; i < npts; ++i) {
    fin >> v[i];
    cout << v[i] << endl;
  }
  return v;
}

void testMatrix() {
  DenseMatrix A = DenseMatrix::random(5, 5);
  cout << "A = \n" << A << endl;
  auto B = A.inv();
  cout << "B = \n" << B << endl;
  cout << "Bt = \n" << B.transposed() << endl;
  cout << "A*B = \n" << A * B << endl;
  cout << "B*A = \n" << B * A << endl;
}

#if 1
void testSparseMatrix() {
  /*
  2 -1 0 0 0
  -1 2 -1 0 0
  0 -1 2 -1 0
  0 0 -1 2 -1
  0 0 0 -1 2
  */
  SparseMatrix M(5, 5, 13);
  M.append(0, 0, 2); M.append(0, 1, -1);
  M.append(1, 0, -1); M.append(1, 1, 2); M.append(1, 2, -1);
  M.append(2, 1, -1); M.append(2, 2, 2); M.append(2, 3, -1);
  M.append(3, 2, -1); M.append(3, 3, 2); M.append(3, 4, -1);
  M.append(4, 3, -1); M.append(4, 4, 2);
  M.append(2, 1, 2);

  auto MtM = M.selfProduct();
  cholmod_print_sparse(MtM, "MtM", global::cm);

  DenseVector b(5);
  for(int i=0;i<5;++i) b(i) = 1.0;

  auto x = M.solve(b, true);
  for (int i = 0; i < x.length(); ++i) cout << x(i) << ' ';
  cout << endl;
  DenseVector b1 = M * x;
  for (int i = 0; i < b1.length(); ++i) cout << b1(i) << ' ';
  cout << endl;
}
#endif

int main(int argc, char *argv[])
{
#if 0
  QApplication a(argc, argv);
  BlendshapeGeneration w;
  w.show();
  return a.exec();
#else
  global::initialize();

  testMatrix();
  testSparseMatrix();

#ifdef __APPLE__
  const string datapath = "/Users/phg/Data/FaceWarehouse_Data_0/";
#endif

#ifdef __linux__
  const string datapath = "/home/phg/Data/FaceWarehouse_Data_0/";
#endif

  BasicMesh m;
  m.load(datapath + "Tester_1/Blendshape/shape_0.obj");

  MeshDeformer deformer;
  deformer.setSource(m);

  vector<int> landmarks = loadLandmarks(datapath+"landmarks_74_new.txt");
  deformer.setLandmarks(landmarks);

  int objidx = 2;
  BasicMesh T;
  T.load(datapath + "Tester_1/TrainingPose/pose_" + to_string(objidx) + ".obj");

  PointCloud lm_points;
  lm_points.points = T.verts.row(landmarks);
  BasicMesh D = deformer.deformWithMesh(T, lm_points);

  D.write("deformed" + to_string(objidx) + ".obj");

  global::finalize();
  return 0;
#endif
}
