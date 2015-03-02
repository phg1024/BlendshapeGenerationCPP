#include "common.h"

#include "blendshapegeneration.h"
#include <QtWidgets/QApplication>

#include "testcases.h"

#include "densematrix.h"
#include "densevector.h"
#include "basicmesh.h"
#include "meshdeformer.h"
#include "meshtransferer.h"
#include "cereswrapper.h"

#include "Geometry/matrix.hpp"
#include "utils.h"

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

  int objidx = 1;
  BasicMesh T;
  T.load(datapath + "Tester_1/TrainingPose/pose_" + to_string(objidx) + ".obj");

  PointCloud lm_points;
  lm_points.points = T.verts.row(landmarks);
  BasicMesh D = deformer.deformWithMesh(T, lm_points, 5);

  D.write("deformed" + to_string(objidx) + ".obj");
}


void deformationTransfer() {
#ifdef __APPLE__
  const string datapath = "/Users/phg/Data/FaceWarehouse_Data_0/";
#endif

#ifdef __linux__
  const string datapath = "/home/phg/Data/FaceWarehouse_Data_0/";
#endif

  BasicMesh S0;
  S0.load(datapath + "Tester_1/Blendshape/shape_0.obj");

  BasicMesh T0;
  T0.load(datapath + "Tester_106/Blendshape/shape_0.obj");

  // use deformation transfer to create an initial set of blendshapes
  MeshTransferer transferer;

  transferer.setSource(S0); // set source and compute deformation gradient
  transferer.setTarget(T0); // set target and compute deformation gradient

  // find the stationary set of verteces
  vector<int> stationary_indices = T0.filterVertices([=](double *v) {
    return v[2] <= -0.45;
  });
  transferer.setStationaryVertices(stationary_indices);

  BasicMesh S;
  S.load(datapath + "Tester_1/Blendshape/shape_22.obj");

  BasicMesh T = transferer.transfer(S);
  T.write("transferred.obj");
}

struct PointResidual {
  PointResidual(double x, double y, double z, int idx, const vector<Array2D<double>> &dB)
    : mx(x), my(y), mz(z), vidx(idx), dB(dB) {}

  template <typename T>
  bool operator()(const T* const alpha, T* residual) const {
    T p[3];
    p[0] = T(0); p[1] = T(0); p[2] = T(0);
    int nshapes = dB.size();
    // compute
    for(int i=0;i<nshapes;++i) {
      double *dBi_ptr = dB[i].rowptr(vidx);
      p[0] += T(dBi_ptr[0]) * alpha[i];
      p[1] += T(dBi_ptr[1]) * alpha[i];
      p[2] += T(dBi_ptr[2]) * alpha[i];
    }
    residual[0] = T(mx) - p[0]; residual[1] = T(my) - p[1]; residual[2] = T(mz) - p[2];
    return true;
  }

private:
  const double mx, my, mz;
  const int vidx;
  const vector<Array2D<double>> &dB;
};

struct PriorResidue {
  PriorResidue(double *prior):mprior(prior){}

  template <typename T>
  bool operator()(const T* const alpha, T* residual) const {
    int nshapes = 46;
    for(int i=0;i<nshapes;++i) residual[i] = T(mprior[i]) - alpha[i];
    return true;
  }
private:
  const double *mprior;
};

Array1D<double> estimateWeights(const BasicMesh &S,
                                const BasicMesh &B0,
                                const vector<Array2D<double>> &dB,
                                const Array1D<double> &w0,  // init value
                                const Array1D<double> &wp,  // prior
                                double w_prior,
                                int itmax) {
  Array1D<double> w = w0;

  Problem problem;

  // add all constraints
  int nverts = S.verts.nrow;
  for(int i=0;i<nverts;++i) {
    double *vS = S.verts.rowptr(i);
    double *vB0 = B0.verts.rowptr(i);
    double dx = vS[0] - vB0[0];
    double dy = vS[1] - vB0[1];
    double dz = vS[2] - vB0[2];

    CostFunction *costfun = new AutoDiffCostFunction<PointResidual, 3, 46>(
          new PointResidual(dx, dy, dz, i, dB));
    problem.AddResidualBlock(costfun, NULL, w.data.get());
  }

  // add prior if necessary
  if( fabs(w_prior) > 1e-6 ) {
    problem.AddResidualBlock(
          new AutoDiffCostFunction<PriorResidue, 46, 46>(new PriorResidue(wp.data.get())),
          NULL, w.data.get());
  }

  cout << "w0 = " << endl;
  cout << w << endl;
  // set the solver options
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;

  return w;
}

vector<BasicMesh> refineBlendShapes(const vector<BasicMesh> &S,
                                          const vector<Array2D<double>> &Sgrad,
                                          const vector<BasicMesh> &B,
                                          const vector<Array1D<double>> &alpha,
                                          double beta, double gamma,
                                          const Array2D<double> prior,
                                          const Array2D<double> w_prior,
                                          const vector<int> stationary_indices) {
  const BasicMesh &B0 = B[0];

  int nfaces = B0.faces.nrow;
  int nverts = B0.verts.nrow;

  int nposes = S.size();
  int nshapes = B.size() - 1;

  // compute the deformation gradients for B0
  Array2D<double> B0grad(nfaces, 9);
  Array1D<double> DB(nfaces);
  for(int i=0;i<nfaces;++i) {
    auto GD = triangleGradient2(B0, i);
    auto B0i_ptr = B0grad.rowptr(i);
    for(int k=0;k<9;++k) B0i_ptr[k] = GD.first(k);
    DB(i) = GD.second;
  }

  int nrows = nposes + nshapes;
  int ncols = nshapes;
  DenseMatrix A0(nrows, ncols);
  // the upper part of A is always the alpha matrix
  for(int i=0;i<nposes;++i) {
    for(int j=0;j<ncols;++j) {
      A0(i, j) = alpha[i](j);
    }
  }

  cout << BLUE << "computing perface gradients." << RESET << endl;
  vector<DenseMatrix> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    if( j % 5000 == 0 ) cout << RED << j << " faces processed." << RESET << endl;
    DenseMatrix A = A0;
    DenseMatrix b(nrows, 9);

    auto B0j_ptr = B0grad.rowptr(j);
    // upper part of b
    for(int i=0;i<nposes;++i) {
      // the gradients of the i-th pose
      auto Sgrad_j = Sgrad[i];
      // the pointer to the j-th face
      auto Sgrad_ij = Sgrad_j.rowptr(j);
      for(int k=0;k<9;++k) b(i, k) = Sgrad_ij[k] - B0j_ptr[k];
    }

    // lower part of A
    for(int i=0;i<nshapes;++i) {
      A(i+nposes, i) = beta * w_prior(i, j);
    }

    // the lower part of b
    int ioffset = j*9;
    for(int i=0;i<nshapes;++i) {
      for(int k=0;k<9;++k) b(i+nposes, k) = beta * w_prior(i, j) * prior(i, ioffset+k);
    }

    // solve the equation A'A\A'b
    DenseMatrix At = A.transposed();
    DenseMatrix AtA = At * A;
    DenseMatrix Atb = At * b;
    M[j] = (AtA.inv()) * Atb;     // nshapes x 9 matrix
  }

  // reconstruct the blendshapes now
  cout << BLUE << "reconstructing blendshapes." << RESET << endl;
  MeshTransferer transferer;
  transferer.setSource(B0);
  transferer.setTarget(B0);
  transferer.setStationaryVertices(stationary_indices);

  vector<BasicMesh> B_new(nshapes);
  for(int i=0;i<nshapes;++i) {
    cout << GREEN << "reconstructing blendshapes " << i << RESET << endl;
    vector<PhGUtils::Matrix3x3d> Bgrad_i(nfaces);
    for(int j=0;j<nfaces;++j) {
      auto &Mj = M[j];
      PhGUtils::Matrix3x3d M0j(B0grad.rowptr(j), false);
      PhGUtils::Matrix3x3d Mij;
      for(int k=0;k<9;++k) Mij(k) = Mj(i, k);
      PhGUtils::Matrix3x3d Bgrad_ij = ((Mij + M0j) * M0j.inv()).transposed();
      for(int k=0;k<9;++k) Bgrad_i[j](k) = Bgrad_ij(k);
    }
    B_new[i] = transferer.transfer(Bgrad_i);
  }

  return B_new;
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

  const bool synthesizeTrainingPoses = true;
  vector<Array1D<double>> alpha_ref(nposes);
  if( synthesizeTrainingPoses ) {
    // estimate the blendshape weights from the input training poses, then use
    // the estimated weights to generate a new set of training poses

    // compute the delta shapes of the renference shapes
    vector<Array2D<double>> dB_ref(nshapes);
    for(int i=0;i<nshapes;++i) {
      dB_ref[i] = B_ref[i+1].verts - B0.verts;
    }

    // estimate the weights of the training poses using the ground truth blendshapes
    for(int i=0;i<nposes;++i) {
      alpha_ref[i] = estimateWeights(S0[i], B0, dB_ref,
                                     Array1D<double>::ones(nshapes) * 0.25,
                                     Array1D<double>::zeros(nshapes),
                                     0.0, 5);
      cout << alpha_ref[i] << endl;
    }

    // use the reference weights to build a set of training poses
    // this set is used as ground truth set
    vector<BasicMesh> Sgen(nposes);
    for(int i=0;i<nposes;++i) {
      // make a copy of B0
      auto Ti = B0;
      for(int j=0;j<nshapes;++j) {
        Ti.verts += alpha_ref[i](j) * dB_ref[j];
      }
      Ti.write("Sgen_"+to_string(i)+".obj");
      //Ti = alignMesh(Ti, S0{i});
      //S_error(iters, i) = sqrt(max(sum((Ti.vertices-S0{i}.vertices).^2, 2)));

      // update the reconstructed mesh
      Sgen[i] = Ti;
    }
    S0 = Sgen;
    cout << "input training set generated." << endl;
  }
  else {
    // fill alpha_ref with zeros
    for(int i=0;i<nposes;++i) alpha_ref[i] = Array1D<double>::random(nshapes);
  }

  // create point clouds from S0
  vector<PointCloud> P(nposes);
  for(int i=0;i<nposes;++i) {
    P[i] = S0[i].samplePoints(4, -0.10);
    P[i].write("P_" + to_string(i) + ".txt");
  }

  vector<BasicMesh> S(nposes);  // meshes reconstructed from point clouds
  // use Laplacian deformation to reconstruct a set of meshes from the sampled
  // point clouds
  MeshDeformer deformer;
  deformer.setSource(B0);
  deformer.setLandmarks(landmarks);
  for(int i=0;i<nposes;++i) {
    PointCloud lm_points;
    lm_points.points = S0[i].verts.row(landmarks);
    S[i] = deformer.deformWithPoints(P[i], lm_points, 2);
    S[i].write("Sinit_" + to_string(i) + ".obj");
  }
  auto S_init = S;
  cout << "initial fitted meshes computed." << endl;

  // find the stationary set of verteces
  vector<int> stationary_indices = B0.filterVertices([=](double *v) {
    return v[2] <= -0.45;
  });
  cout << "stationary vertices = " << stationary_indices.size() << endl;

  cout << BLUE << "creating initial set of blendshapes using deformation transfer." << RESET << endl;
  // use deformation transfer to create an initial set of blendshapes
  MeshTransferer transferer;
  transferer.setSource(A0); // set source and compute deformation gradient
  transferer.setTarget(B0); // set target and compute deformation gradient
  transferer.setStationaryVertices(stationary_indices);

  for(int i=1;i<=nshapes;++i) {
    cout << GREEN << "creating shape " << i << RESET << endl;
    B[i] = transferer.transfer(A[i]);
    B[i].write("Binit_" + to_string(i) + ".obj");
  }
  auto B_init = B;
  cout << BLUE << "initial set of blendshapes created." << RESET << endl;

  // compute deformation gradient prior from the template mesh
  cout << BLUE << "computing priors." << RESET << endl;
  int nfaces = A0.faces.nrow;

  Array2D<double> MB0 = Array2D<double>::zeros(nfaces, 9);
  Array2D<double> MA0 = Array2D<double>::zeros(nfaces, 9);

  for(int j=0;j<nfaces;++j) {
    auto MB0j = triangleGradient(B0, j);
    auto MB0j_ptr = MB0.rowptr(j);
    for(int k=0;k<9;++k) MB0j_ptr[k] = MB0j(k);
    auto MA0j = triangleGradient(A0, j);
    auto MA0j_ptr = MA0.rowptr(j);
    for(int k=0;k<9;++k) MA0j_ptr[k] = MA0j(k);
  }

  double kappa = 0.1;
  Array2D<double> prior = Array2D<double>::zeros(nshapes, 9*nfaces);
  Array2D<double> w_prior = Array2D<double>::zeros(nshapes, nfaces);
  for(int i=0;i<nshapes;++i) {
    // prior for shape i
    auto &Ai = A[i+1];
    Array2D<double> Pi = Array2D<double>::zeros(nfaces, 9);
    auto w_prior_i = w_prior.rowptr(i);
    for(int j=0;j<nfaces;++j) {
      auto MAij = triangleGradient(Ai, j);
      auto MA0j_ptr = MA0.rowptr(j);
      // create a 3x3 matrix from MA0j_ptr
      auto MA0j = PhGUtils::Matrix3x3d(MA0j_ptr, false);
      auto GA0Ai = MAij * (MA0j.inv());
      auto MB0j_ptr = MB0.rowptr(j);
      auto MB0j = PhGUtils::Matrix3x3d(MB0j_ptr, false);
      auto Pij = GA0Ai * MB0j - MB0j;
      double MAij_norm = (MAij-MA0j).norm();
      w_prior_i[j] = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0) * 100;

      auto Pij_ptr = Pi.rowptr(j);
      for(int k=0;k<9;++k) Pij_ptr[k] = Pij(k);
    }
    auto prior_i = prior.rowptr(i);
    // prior(i,:) = Pi;
    memcpy(prior_i, Pi.data.get(), sizeof(double)*nfaces*9);
  }

  ofstream fprior("prior.txt");
  fprior<<prior;
  fprior.close();

  ofstream fwprior("w_prior.txt");
  fwprior<<w_prior;
  fwprior.close();

  cout << BLUE << "priors computed." << RESET << endl;

  // compute the delta shapes
  vector<Array2D<double>> dB(nshapes);
  for(int i=0;i<nshapes;++i) {
    dB[i] = B[i+1].verts - B0.verts;
  }

  // estimate initial set of blendshape weights
  vector<Array1D<double>> alpha(nposes);
  for(int i=0;i<nposes;++i) {
    alpha[i] = estimateWeights(S[i], B0, dB,
                               Array1D<double>::random(nshapes),
                               Array1D<double>::zeros(nshapes),
                               0.0, 5);
  }
  auto alpha_init = alpha;

  cout << RED << "initialization done." << RESET << endl;

  // reset the parameters
  B = B_init;
  S = S_init;
  alpha = alpha_init;

  // main blendshape refinement loop

  // compute deformation gradients for S
  // Note: this deformation gradient cannot be used to call
  // MeshTransferer::transfer directly. See this function for
  // details
  vector<Array2D<double>> Sgrad(nposes);
  for(int i=0;i<nposes;++i) {
    Sgrad[i] = Array2D<double>::zeros(nfaces, 9);
    for(int j=0;j<nfaces;++j) {
      // assign the reshaped gradient to the j-th row of Sgrad[i]
      auto Sij = triangleGradient(S[i], j);
      auto Sij_ptr = Sgrad[i].rowptr(j);
      for(int k=0;k<9;++k) Sij_ptr[k] = Sij(k);
    }
  }

#if 0   // PASSED Verification

  // verify the deformation gradients are computed correctly
  cout << "Creating meshes for verification." << endl;
  MeshTransferer transer;
  transer.setSource(S.front());
  transer.setTarget(S.front());
  transer.setStationaryVertices(stationary_indices);
  for(int i=0;i<nposes;++i) {
    // assemble the set of deformation gradient
    vector<PhGUtils::Matrix3x3d> grads;
    for(int j=0;j<nfaces;++j) grads.push_back(
                (PhGUtils::Matrix3x3d(Sgrad[i].rowptr(j))
                * PhGUtils::Matrix3x3d(Sgrad[0].rowptr(j)).inv()).transposed()
                );
    BasicMesh Si = transer.transfer(grads);
    Si.write("Sinit_verify_" + to_string(i) + ".obj");
  }
  cout << "done." << endl;
  return;
#endif

  // refine blendshapes as well as blendshape weights
  bool converged = false;
  double ALPHA_THRES = 1e-6;
  double B_THRES = 1e-6;
  double beta_max = 0.5, beta_min = 0.1;
  double gamma_max = 0.01, gamma_min = 0.01;
  double eta_max = 10.0, eta_min = 1.0;
  int iters = 0;
  const int maxIters = 5;
  DenseMatrix B_error = DenseMatrix::zeros(maxIters, nshapes);
  DenseMatrix S_error = DenseMatrix::zeros(maxIters, nposes);
  while( !converged && iters < maxIters ) {
    cout << "iteration " << iters << " ..." << endl;
    converged = true;
    ++iters;

    // refine blendshapes
    double beta = sqrt(iters/maxIters) * (beta_min - beta_max) + beta_max;
    double gamma = gamma_max + iters/maxIters*(gamma_min-gamma_max);
    double eta = eta_max + iters/maxIters*(eta_min-eta_max);

    // B_new is a set of new blendshapes
    auto B_new = refineBlendShapes(S, Sgrad, B, alpha, beta, gamma, prior, w_prior, stationary_indices);

    //B_norm = zeros(1, nshapes);
    for(int i=0;i<nshapes;++i) {
      //B_norm(i) = norm(B[i+1].vertices-B_new[i], 2);
      B[i+1].verts = B_new[i].verts;
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
        Ti.verts += alpha[i](j) * dB[j];
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
      S[i] = deformer.deformWithPoints(P[i], lm_points, 2);
    }

    // compute deformation gradients for S
    for(int i=0;i<nposes;++i) {
      Sgrad[i] = Array2D<double>::zeros(nfaces, 9);
      for(int j=0;j<nfaces;++j) {
        // assign the reshaped gradient to the j-th row of Sgrad[i]
        auto Sij = triangleGradient(S[i], j);
        auto Sij_ptr = Sgrad[i].rowptr(j);
        for(int k=0;k<9;++k) Sij_ptr[k] = Sij(k);
      }
    }
  }

  // write out the blendshapes
  for(int i=0;i<nshapes+1;++i) {
    B[i].write("B_"+to_string(i)+".obj");
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
  TestCases::testMatrix();
  TestCases::testSaprseMatrix();
  TestCases::testCeres();

  return 0;
#else

//  deformationTransfer();
//  return 0;

//  laplacianDeformation();
//  return 0;

  blendShapeGeneration();

  global::finalize();
  return 0;
#endif

#endif
}
