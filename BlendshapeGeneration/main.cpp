#include "common.h"

#include "blendshapegeneration.h"
#include <QtWidgets/QApplication>

#include "densematrix.h"
#include "densevector.h"
#include "basicmesh.h"
#include "meshdeformer.h"
#include "meshtransferer.h"
#include "cereswrapper.h"

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

void testCeres() {

  // The variable to solve for with its initial value.
  double initial_x = 5.0;
  double x = initial_x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, &x);

  // Run the solver!
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
}

void laplacianDeformation() {
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
}

void blendShapeGeneration() {
  // load the meshes
  string datapath = "/home/phg/Data/FaceWarehouse_Data_0/";
  string A_path = datapath + "Tester_1/Blendshape/";
  string B_path = datapath + "Tester_106/Blendshape/";
  string S_path = datapath + "Tester_106/TrainingPose/";

  const int nshapes = 46; // 46 in total (1..46)
  const int nposes = 19; // 20 in total (0..19)

  vector<BasicMesh> A(nshapes+1);
  vector<BasicMesh> B(nshapes+1);
  vector<BasicMesh> B_ref(nshapes+1);
  vector<BasicMesh> S0(nposes);             // ground truth training meshes

  // load the landmarks
  vector<int> landmarks = loadLandmarks(datapath+"landmarks_74_new.txt");

  // load the template blendshapes and ground truth blendshapes
  for(int i=0;i<=nshapes;++i) {
    A[i].load(A_path + "shape_" + to_string(i) + ".obj");
    B_ref[i].load(B_path + "shape_" + to_string(i) + ".obj");
  }

  B[0] = B_ref[0];

  // reference shapes for convenience
  auto& A0 = A[0];
  auto& B0 = B[0];

  // load the training poses
  for(int i=0;i<nposes;++i){
    S0[i].load(S_path + "pose_" + to_string(i) + ".obj");
  }

  const bool synthesizeTrainingPoses = false;
  if( synthesizeTrainingPoses ) {
    // estimate the blendshape weights from the input training poses, then use
    // the estimated weights to generate a new set of training poses
  }

  // create point clouds from S0
  vector<PointCloud> P(nposes);
  for(int i=0;i<nposes;++i) {
    P[i] = samplePointClouds(S0);
  }

  vector<BasicMesh> S(nposes);  // meshes reconstructed from point clouds
  // use Laplacian deformation to reconstruct a set of meshes from the sampled
  // point clouds
  for(int i=0;i<nposes;++i) {
    MeshDeformer deformer;
    deformer.setSource(S0[i]);
    deformer.setLandmarks(landmarks);
    PointCloud lm_points;
    lm_points.points = S0[i].verts.row(landmarks);
    S[i] = deformer.deformWithPoints(P[i], lm_points);
  }
  auto S_init = S;

  // find the stationary set of verteces
  // stationary_indices = find(B{1}.vertices(:,3)<-0.45);

  // use deformation transfer to create an initial set of blendshapes
  MeshTransferer transferer;
  transferer.setSource(A0); // set source and compute deformation gradient
  transferer.setTarget(B0); // set target and compute deformation gradient

  for(int i=1;i<nshapes;++i) {
    B[i] = transferer.transfer(A[i]);
  }
  auto B_init = B;

  // compute deformation gradient prior from the template mesh


  // compute the delta shapes
  vector<PointCloud> dB(nshapes);
  for(int i=0;i<nshapes;++i) {
    dB[i] = B[i+1].verts - B0.verts;
  }

  // estimate initial set of blendshape weights
  vector<Array1D<double>> alpha(nposes);
  for(int i=0;i<nposes;++i) {
    alpha[i] = estimateWeights(S, B0, dB, Array1D::zeros(nshapes), 0.0, 5, true);
  }
  auto alpha_init = alpha;

  cout << "initialization done." << endl;

  // reset the parameters
  B = B_init;
  S = S_init;
  alpha = alpha_init;

  // main blendshape refinement loop

  // compute deformation gradients for S
  vector<DenseMatrix> Sgrad(nfaces);
  for(int i=0;i<nfaces;++i) {
    Sgrad[i] = DenseMatrix::zeros(nposes, 9);
    for(int j=0;j<nposes;++j) {
      // assign the reshaped gradient to the i-th row of Sgrad
      // Sij = triangleGradient(S{i}, j);
      // Smat(i,:) = reshape(Sij, 1, 9);

      // FIXME: not implemented yet
    }
  }

  // refine blendshapes as well as blendshape weights
  bool converged = false;
  double ALPHA_THRES = 1e-6;
  double B_THRES = 1e-6;
  double beta_max = 0.5; beta_min = 0.1;
  double gamma_max = 0.01; gamma_min = 0.01;
  double eta_max = 10.0; eta_min = 1.0;
  int iters = 0;
  const int maxIters = 10;
  DenseMatrix B_error = DenseMatrix::zeros(maxIters, nshapes);
  DenseMatrix S_error = DenseMatrix::zeros(maxIters, nposes);
  while( !converged && iters < maxIters ) {
      cout << "iteration " << iters << " ..." << endl;
      converged = true;
      ++iters;

      // refine blendshapes
      beta = sqrt(iters/maxIters) * (beta_min - beta_max) + beta_max;
      gamma = gamma_max + iters/maxIters*(gamma_min-gamma_max);
      eta = eta_max + iters/maxIters*(eta_min-eta_max);

      // B_new is a set of point clouds
      auto B_new = refineBlendShapes(S, Sgrad, B, alpha, beta, gamma, prior, w_prior, stationary_indices);

      B_norm = zeros(1, nshapes);
      for(int i=0;i<nshapes;++i) {
          //B_norm(i) = norm(B[i+1].vertices-B_new[i], 2);
          B[i+1].vertices = B_new[i];
          //B_error(iters, i) = sqrt(max(sum((B{i+1}.vertices-B_ref{i+1}.vertices).^2, 2)));
          //B{i+1} = alignMesh(B{i+1}, B{1}, stationary_indices);
      }
      //fprintf('max(B_error) = %.6f\n', max(B_error(iters, :)));
      //converged = converged & (max(B_norm) < B_THRES);

      // update delta shapes
      for(int i=0;i<nshapes;++i) {
        dB[i] = B[i+1].verts - B0.verts;
      }

      // update weights
      vector<Array1D<double>> alpha_new(nposes);
      for(int i=0;i<nposes;++i) {
        alpha_new[i] = estimateWeights(S[i], B0, dB, alpha[i], alpha_ref[i], eta, 2);
      }

      //alpha_norm = norm(cell2mat(alpha) - cell2mat(alpha_new));
      //disp(alpha_norm);
      alpha = alpha_new;
      //converged = converged & (alpha_norm < ALPHA_THRES);

      for(int i=0;i<nposes;++i) {
        // make a copy of B0
        auto Ti = B0;
          for(int j=0;j<nshapes;++j) {
              Ti.vertices += alpha[i](j) * dB[j];
          }
          //Ti = alignMesh(Ti, S0{i});
          //S_error(iters, i) = sqrt(max(sum((Ti.vertices-S0{i}.vertices).^2, 2)));

          // update the reconstructed mesh
          S[i] = Ti;
      }
      //fprintf('Emax = %.6f\tEmean = %.6f\n', max(S_error(iters,:)), mean(S_error(iters,:)));


      // The reconstructed mesh are updated using the new set of blendshape
      // weights, need to use laplacian deformation to refine them again
      for(int i=0;i<nposes;++i) {
        MeshDeformer deformer;
        deformer.setSource(S[i]);
        deformer.setLandmarks(landmarks);
        PointCloud lm_points;
        lm_points.points = S0[i].verts.row(landmarks);
        S[i] = deformer.deformWithPoints(P[i], lm_points);
      }

      // compute deformation gradients for S
      // compute deformation gradients for S
      for(int i=0;i<nfaces;++i) {
        Sgrad[i] = DenseMatrix::zeros(nposes, 9);
        for(int j=0;j<nposes;++j) {
          // assign the reshaped gradient to the i-th row of Sgrad
          // Sij = triangleGradient(S{i}, j);
          // Smat(i,:) = reshape(Sij, 1, 9);

          // FIXME: not implemented yet
        }
      }
  }
}

int main(int argc, char *argv[])
{
#if 0
  QApplication a(argc, argv);
  BlendshapeGeneration w;
  w.show();
  return a.exec();
#else
  google::InitGoogleLogging(argv[0]);

  global::initialize();

#if RUN_TESTS
  testMatrix();
  testSparseMatrix();
  testCeres();

  return 0;
#else

  //laplacianDeformation();
  blendShapeGeneration();

  global::finalize();
  return 0;
#endif

#endif
}
