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
#include <MultilinearReconstruction/statsutils.h>

#include "blendshaperefiner.h"
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

#if 0
struct ImageBundle {
  ImageBundle() {}
  ImageBundle(const QImage& image, const vector<Constraint2D>& points, const ReconstructionResult& params)
    : image(image), points(points), params(params) {}
  QImage image;
  vector<Constraint2D> points;
  ReconstructionResult params;
};
#endif

struct PointResidual {
  PointResidual(double x, double y, double z, int idx, const vector<MatrixX3d> &dB)
    : mx(x), my(y), mz(z), vidx(idx), dB(dB) {}

  template <typename T>
  bool operator()(const T* const alpha, T* residual) const {
    T p[3];
    p[0] = T(0); p[1] = T(0); p[2] = T(0);
    int nshapes = dB.size();
    // compute
    for(int i=0;i<nshapes;++i) {
      auto dBi_ptr = dB[i].row(vidx);
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
  const vector<MatrixX3d> &dB;
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
                                const vector<MatrixX3d> &dB,
                                const Array1D<double> &w0,  // init value
                                const Array1D<double> &wp,  // prior
                                double w_prior,
                                int itmax) {
  Array1D<double> w = w0;

  Problem problem;

  // add all constraints
  int nverts = S.NumVertices();
  for(int i=0;i<nverts;++i) {
    auto vS = S.vertex(i);
    auto vB0 = B0.vertex(i);
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

  options.num_threads = 8;
  options.num_linear_solver_threads = 8;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

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

  int nfaces = B0.NumFaces();
  int nverts = B0.NumVertices();

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

#define USE_LONG_VECTOR_FORM 1
#if USE_LONG_VECTOR_FORM
  using Tripletd = Eigen::Triplet<double>;
  using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  vector<Tripletd> Adata_coeffs;

  // fill the upper part of A
  for(int i=0;i<nposes;++i) {
    int row_offset = i * 9;
    for(int j=0;j<nshapes;++j) {
      int col_offset = j * 9;
      for(int k=0;k<9;++k) {
        Adata_coeffs.push_back(Tripletd(row_offset+k, col_offset+k, alpha[i](j)));
      }
    }
  }

  int nrows = (nposes+nshapes) * 9;
  int ncols = nshapes * 9;

  vector<MatrixXd> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    if( j % 5000 == 0 ) ColorStream(ColorOutput::Red) << j << " faces processed.";

    VectorXd b(nrows);

    auto B0j_ptr = B0grad.rowptr(j);
    // upper part of b
    for(int i=0;i<nposes;++i) {
      // the gradients of the i-th pose
      auto Sgrad_j = Sgrad[i];
      // the pointer to the j-th face
      auto Sgrad_ij = Sgrad_j.rowptr(j);
      for(int k=0;k<9;++k) b(i*9+k) = Sgrad_ij[k] - B0j_ptr[k];
    }

    // lower part of A
    vector<Tripletd> A_coeffs = Adata_coeffs;
    for(int i=0;i<nshapes;++i) {
      int row_offset = (i+nposes)*9;
      int col_offset = i * 9;
      for(int k=0;k<9;++k) {
        A_coeffs.push_back(Tripletd(row_offset + k, col_offset + k, beta * w_prior(i, j)));
      }
    }

    // the lower part of b
    int ioffset = j*9;
    for(int i=0;i<nshapes;++i) {
      for(int k=0;k<9;++k) b((i+nposes)*9+k) = beta * w_prior(i, j) * prior(i, ioffset+k);
    }

    // solve the equation A'A\A'b
    Eigen::SparseMatrix<double> A(nrows, ncols);
    A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
    A.makeCompressed();

    Eigen::SparseMatrix<double> AtA = (A.transpose() * A).pruned();

    const double epsilon = 0.0;//1e-9;
    Eigen::SparseMatrix<double> eye(ncols, ncols);
    for(int j=0;j<ncols;++j) eye.insert(j, j) = epsilon;
    AtA += eye;

    CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);
    if(solver.info()!=Success) {
      cerr << "Failed to decompose matrix A." << endl;
      exit(-1);
    }

    VectorXd Atb = A.transpose() * b;
    VectorXd Mv = solver.solve(Atb);
    if(solver.info()!=Success) {
      cerr << "Failed to solve A\\b." << endl;
      exit(-1);
    }

    // convert Mv to nshapes x 9 matrix
    M[j] = MatrixXd(nshapes, 9);
    for(int i=0;i<nshapes;++i) {
      M[j].row(i) = Mv.middleRows(i*9, 9).transpose();
    }
  }
#else
  int nrows = nposes + nshapes;
  int ncols = nshapes;
  MatrixXd A0(nrows, ncols);
  // the upper part of A is always the alpha matrix
  for(int i=0;i<nposes;++i) {
    for(int j=0;j<ncols;++j) {
      A0(i, j) = alpha[i](j);
    }
  }

  ColorStream(ColorOutput::Blue) << "computing perface gradients.";
  vector<MatrixXd> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    if( j % 5000 == 0 ) ColorStream(ColorOutput::Red) << j << " faces processed.";
    MatrixXd A = A0;
    MatrixXd b(nrows, 9);

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
    M[j] = (A.transpose() * A).ldlt().solve(A.transpose() * b);     // nshapes x 9 matrix
  }
#endif

  // reconstruct the blendshapes now
  ColorStream(ColorOutput::Blue) << "reconstructing blendshapes.";
  MeshTransferer transferer;
  transferer.setSource(B0);
  transferer.setTarget(B0);
  transferer.setStationaryVertices(stationary_indices);

  vector<BasicMesh> B_new(nshapes);
  for(int i=0;i<nshapes;++i) {
    ColorStream(ColorOutput::Green)<< "reconstructing blendshapes " << i;
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

pair<VectorXd, BasicMesh> estimateNeutralFace(const BasicMesh& target) {
  MultilinearModel model("/home/phg/Data/Multilinear/blendshape_core.tensor");
  MultilinearModelPrior model_prior;
  model_prior.load("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");

  // Just the neutral expression
  VectorXd w_exp_FACS = VectorXd::Zero(47);
  w_exp_FACS(0) = 1.0;

  VectorXd w_exp = (w_exp_FACS.transpose() * model_prior.Uexp).eval();

  model.UpdateTM1(w_exp);

  Tensor2 tm1 = model.GetTM1();
  // tm1_mat^T * w_exp -> verts
  MatrixXd tm1_mat = tm1.GetData();
  cout << tm1_mat.rows() << 'x' << tm1_mat.cols() << endl;

  int nverts = tm1.cols() / 3;
  VectorXd target_verts(tm1_mat.cols());
  for(int i=0;i<nverts;++i) {
    auto Vi = target.vertex(i);
    target_verts(3*i) = Vi[0];
    target_verts(3*i+1) = Vi[1];
    target_verts(3*i+2) = Vi[2];
  }

  VectorXd wid = tm1_mat.transpose().jacobiSvd(ComputeThinU | ComputeThinV).solve(target_verts);

  // perturb a little
  double wid_0 = wid(0);
  wid = StatsUtils::perturb(wid, 0.1);
  wid(0) = wid_0;

  VectorXd verts = (wid.transpose() * tm1_mat).transpose();
  BasicMesh res = target;

  for(int i=0;i<nverts;++i) {
    res.set_vertex(i, Vector3d(verts(i*3), verts(i*3+1), verts(i*3+2)));
  }
  res.Write("Bgen.obj");

  target.Write("Btgt.obj");
  return make_pair(wid, res);
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
  vector<int> landmarks = LoadIndices(datapath+"landmarks_74_new.txt");

  // load the template blendshapes and ground truth blendshapes
  for(int i=0;i<=nshapes;++i) {
    A[i].LoadOBJMesh(A_path + "shape_" + to_string(i) + ".obj");
    B_ref[i].LoadOBJMesh(B_path + "shape_" + to_string(i) + ".obj");
  }

  // Estimate a neutral face using the multi-linear model
  VectorXd w_id;
  tie(w_id, B[0]) = estimateNeutralFace(B_ref[0]);

  // reference shapes for convenience
  auto& A0 = A[0];
  auto& B0 = B[0];

  // load the training poses
  for(int i=0;i<nposes;++i){
    S0[i].LoadOBJMesh(S_path + "pose_" + to_string(i) + ".obj");
  }

  const bool synthesizeTrainingPoses = true;
  vector<Array1D<double>> alpha_ref(nposes);
  if( synthesizeTrainingPoses ) {
    // estimate the blendshape weights from the input training poses, then use
    // the estimated weights to generate a new set of training poses

    // compute the delta shapes of the renference shapes
    vector<MatrixX3d> dB_ref(nshapes);
    for(int i=0;i<nshapes;++i) {
      dB_ref[i] = B_ref[i+1].vertices() - B_ref[0].vertices();
    }

    // estimate the weights of the training poses using the ground truth blendshapes
    for(int i=0;i<nposes;++i) {
      alpha_ref[i] = estimateWeights(S0[i], B_ref[0], dB_ref,
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
      auto Ti = B_ref[0];
      for(int j=0;j<nshapes;++j) {
        Ti.vertices() += alpha_ref[i](j) * dB_ref[j];
      }
      Ti.Write("Sgen_"+to_string(i)+".obj");
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

  // create point clouds from Sgen
  vector<MatrixX3d> P(nposes);
  for(int i=0;i<nposes;++i) {
    P[i] = S0[i].samplePoints(4, -0.10);
    //P[i].Write("P_" + to_string(i) + ".txt");
  }

#if 0
  vector<BasicMesh> S(nposes);  // meshes reconstructed from point clouds
  // use Laplacian deformation to reconstruct a set of meshes from the sampled
  // point clouds
  MeshDeformer deformer;
  deformer.setSource(B0);
  deformer.setLandmarks(landmarks);
  for(int i=0;i<nposes;++i) {
    PointCloud lm_points;

    lm_points.points.resize(landmarks.size(), 3);
    for(int j=0;j<landmarks.size();++j) {
      auto vj = S0[i].vertex(landmarks[j]);
      lm_points.points(j, 0) = vj[0];
      lm_points.points(j, 1) = vj[1];
      lm_points.points(j, 2) = vj[2];
    }

    S[i] = deformer.deformWithPoints(P[i], lm_points, 5);
    S[i].Write("Sinit_" + to_string(i) + ".obj");
  }
  auto S_init = S;
  cout << "initial fitted meshes computed." << endl;
#else
  // Just use the synthesized shapes as ground truth
  vector<BasicMesh> S = S0;
  auto S_init = S;
#endif

  // find the stationary set of verteces
  vector<int> stationary_indices = B0.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  });
  cout << "stationary vertices = " << stationary_indices.size() << endl;

  ColorStream(ColorOutput::Blue) << "creating initial set of blendshapes using deformation transfer.";
  // use deformation transfer to create an initial set of blendshapes
  MeshTransferer transferer;
  transferer.setSource(A0); // set source and compute deformation gradient
  transferer.setTarget(B0); // set target and compute deformation gradient
  transferer.setStationaryVertices(stationary_indices);

  for(int i=1;i<=nshapes;++i) {
    ColorStream(ColorOutput::Green)<< "creating shape " << i;
    B[i] = transferer.transfer(A[i]);
    B[i].Write("Binit_" + to_string(i) + ".obj");
  }
  auto B_init = B;
  ColorStream(ColorOutput::Blue)<< "initial set of blendshapes created.";

  // XXX Need to refer to the blendshape refiner and change the code below to a
  // full optimization of the entire set of blendshapes

  // compute deformation gradient prior from the template mesh
  ColorStream(ColorOutput::Blue)<< "computing priors.";
  int nfaces = A0.NumFaces();

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
      w_prior_i[j] = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0);

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

  ColorStream(ColorOutput::Blue)<< "priors computed.";

  // compute the delta shapes
  vector<MatrixX3d> dB(nshapes);
  for(int i=0;i<nshapes;++i) {
    dB[i] = B[i+1].vertices() - B0.vertices();
  }

  // estimate initial set of blendshape weights
  vector<Array1D<double>> alpha(nposes);
  for(int i=0;i<nposes;++i) {
    alpha[i] = estimateWeights(S[i], B0, dB,
                               //Array1D<double>::random(nshapes),
                               Array1D<double>::ones(nshapes)*0.25,
                               Array1D<double>::zeros(nshapes),
                               0.0, 5);
  }
  auto alpha_init = alpha;

  ColorStream(ColorOutput::Red)<< "initialization done.";

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
  double beta_max = 0.05, beta_min = 0.01;
  double gamma_max = 0.01, gamma_min = 0.01;
  double eta_max = 1.0, eta_min = 0.1;
  int iters = 0;
  const int maxIters = 5;
  MatrixXd B_error = MatrixXd::Zero(maxIters, nshapes);
  MatrixXd S_error = MatrixXd::Zero(maxIters, nposes);
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

    VectorXd B_norm(nshapes);
    double B_norm_max = 0;
    for(int i=0;i<nshapes;++i) {
      auto Bdiff = B[i+1].vertices() - B_new[i].vertices();
      B_norm(i) = Bdiff.norm();
      B_norm_max = max(B_norm_max, B_norm(i));
      B[i+1].vertices() = B_new[i].vertices();
      //B{i+1} = alignMesh(B{i+1}, B{1}, stationary_indices);
    }
    ColorStream(ColorOutput::Green)<< "Bnorm = " << B_norm.transpose();
    converged = converged & (B_norm_max < B_THRES);

    // update delta shapes
    for(int i=0;i<nshapes;++i) {
      dB[i] = B[i+1].vertices() - B0.vertices();
    }

    // update weights
    VectorXd alpha_norm(nposes);
    double alpha_norm_max = 0;
    vector<Array1D<double>> alpha_new(nposes);
    for(int i=0;i<nposes;++i) {
      alpha_new[i] = estimateWeights(S[i], B0, dB, alpha[i], alpha_ref[i], eta, 2);
      auto alpha_diff = alpha_new[i] - alpha[i];
      alpha_norm(i) = cblas_dnrm2(alpha_diff.nrow, alpha_diff.data.get(), 1);
      alpha_norm_max = max(alpha_norm_max, alpha_norm(i));
    }
    alpha = alpha_new;
    ColorStream(ColorOutput::Green)<< "norm(alpha) = " << alpha_norm.transpose();
    converged = converged & (alpha_norm_max < ALPHA_THRES);

    for(int i=0;i<nposes;++i) {
      // make a copy of B0
      auto Ti = B0;
      for(int j=0;j<nshapes;++j) {
        Ti.vertices() += alpha[i](j) * dB[j];
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
      lm_points.points.resize(landmarks.size(), 3);
      for(int j=0;j<landmarks.size();++j) {
        auto vj = S0[i].vertex(landmarks[j]);
        lm_points.points(j, 0) = vj[0];
        lm_points.points(j, 1) = vj[1];
        lm_points.points(j, 2) = vj[2];
      }
      S[i] = deformer.deformWithPoints(P[i], lm_points, 5);
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
    B[i].Write("B_"+to_string(i)+".obj");
  }

  // write out the blendshapes
  for(int i=0;i<nposes;++i) {
    S[i].Write("S_"+to_string(i)+".obj");
  }
}

void blendShapeGeneration_pointcloud(
  const string& source_path,
  bool subdivision) {
  BlendshapeRefiner refiner(
    json{
        {"use_init_blendshapes", false},
        {"subdivision", subdivision},
        {"blendshapes_subdivided", false}
      });
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(source_path);
  refiner.SetPointCloudsPath(source_path + "/SFS");
  refiner.SetInputBlendshapesPath("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/");
  refiner.SetBlendshapesPath(source_path + "/blendshapes");

  refiner.LoadSelectionFile("selection.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputPointClouds();

  refiner.Refine();
}

void blendShapeGeneration_pointcloud_blendshapes(
  const string& source_path,
  const string& recon_path,
  const string& point_clouds_path,
  const string& input_blendshapes_path,
  const string& blendshapes_path,
  bool subdivision,
  bool blendshapes_subdivided
) {
  BlendshapeRefiner refiner(
    json{
      {"use_init_blendshapes", true},
      {"subdivision", subdivision},
      {"blendshapes_subdivided", blendshapes_subdivided}
    }
  );
  refiner.SetBlendshapeCount(46);
  refiner.LoadTemplateMeshes("/home/phg/Data/FaceWarehouse_Data_0/Tester_1/Blendshape/", "shape_");

  refiner.SetResourcesPath(source_path);
  refiner.SetReconstructionsPath(recon_path);
  refiner.SetPointCloudsPath(point_clouds_path);
  refiner.SetInputBlendshapesPath(input_blendshapes_path);
  refiner.SetBlendshapesPath(blendshapes_path);

  refiner.LoadSelectionFile("selection.txt");
  refiner.LoadInputReconstructionResults("settings.txt");
  refiner.LoadInputPointClouds();

  refiner.Refine();
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
  google::InitGoogleLogging(argv[0]);

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help messages")
    ("oldfasion", "Generate blendshapes the old way.")
    ("pointclouds", "Generate blendshapes from point clouds")
    ("pointclouds_with_init_shapes", "Generate blendshapes from point clouds and a set of initial blendshapes")
    ("ebfr", "Generate blendshapes using example based facial rigging method")
    ("repo_path", po::value<string>(), "Path to images repo.")
    ("recon_path", po::value<string>(), "Path to large scale reconstruction results")
    ("pointclouds_path", po::value<string>(), "Path to input point clouds")
    ("init_blendshapes_path", po::value<string>(), "Path to initial blendshapes")
    ("blendshapes_path", po::value<string>(), "Path to output blendshapes")
    ("subdivided", "Indicate the input blendshapes are subdivided")
    ("subdivision", "Enable subdivision")
    ("ref_mesh", po::value<string>(), "Reference mesh for distance computation")
    ("mesh", po::value<string>(), "Mesh to visualize")
    ("vis", "Visualize blendshape mesh")
    ("silent", "Silent visualization using offscreen drawing")
    ("save,s", po::value<string>(), "Save the result to a file");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help")) {
      cout << desc << endl;
      return 1;
    }

    if(vm.count("oldfasion")) {
      throw runtime_error("This is no longer supported.");
      blendShapeGeneration();
    } else if(vm.count("ebfr")) {
      throw runtime_error("This is no longer supported.");
      blendShapeGeneration_pointcloud_EBFR();
    } else if (vm.count("pointclouds")) {
      if(vm.count("repo_path")) {
        blendShapeGeneration_pointcloud(vm["repo_path"].as<string>(),
                                        vm.count("subdivision"));
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
          vm.count("subdivided")
        );
      } else {
        throw po::error("Need to specify repo_path, recon_path, pointclouds_path, init_blendshapes_path, blendshapes_path");
      }
    } else if(vm.count("vis")) {
      bool save_result = vm.count("save");
      bool compare_mode = vm.count("ref_mesh");

      string input_mesh_file;
      if(vm.count("mesh")) {
        input_mesh_file = vm["mesh"].as<string>();
      } else {
        throw po::error("Need to specify mesh");
      }

      QApplication a(argc, argv);
      BlendshapeGeneration w(vm.count("silent"));
      if(!vm.count("silent")) w.show();

      if(compare_mode) {
        string ref_mesh_file = vm["ref_mesh"].as<string>();

        w.LoadMeshes(input_mesh_file, ref_mesh_file);
        w.setWindowTitle(input_mesh_file.c_str());
      } else {
        w.LoadMesh(input_mesh_file);
        w.setWindowTitle(input_mesh_file.c_str());
      }

      if(save_result) {
        w.repaint();
        for(int i=0;i<10;++i)
          qApp->processEvents();

        w.Save(vm["save"].as<string>());
        qApp->processEvents();
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
