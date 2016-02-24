#include "blendshaperefiner.h"
#include "cereswrapper.h"
#include "meshdeformer.h"
#include "meshtransferer.h"
#include "triangle_gradient.h"

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
    A[i].LoadOBJMesh(path + basename + to_string(i) + ".obj");
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
    S[i].Write("S0_" + to_string(i) + ".obj");
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

  Binit[0].Write("Binit_0.obj");

  // Deformation transfer to obtain all other blendshapes
  auto& B0 = Binit[0];
  vector<int> stationary_indices = B0.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  });
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

  // Verify the blendshape and the initial weights are good
  for(int i=0;i<num_poses;++i) {
    // make a copy of B0
    auto Ti = B[0];
    for(int j=0;j<num_shapes;++j) {
      Ti.vertices() += image_bundles[i].params.params_model.Wexp_FACS(j) * (B[j].vertices() - B[0].vertices());
    }
    Ti.ComputeNormals();
    Ti.Write("S0_verify_" + std::to_string(i) + ".obj");
  }
}

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
  PriorResidue(const double *prior):mprior(prior){}

  template <typename T>
  bool operator()(const T* const alpha, T* residual) const {
    int nshapes = 46;
    for(int i=0;i<nshapes;++i) residual[i] = T(mprior[i]) - alpha[i];
    return true;
  }
private:
  const double *mprior;
};

VectorXd BlendshapeRefiner::EstimateWeights(const BasicMesh &S, const BasicMesh &B0, const vector <MatrixX3d> &dB,
                                            const VectorXd &w0, const VectorXd &wp, double w_prior, int itmax) {
  VectorXd w = w0;

  Problem problem;

  // add all constraints
  const int num_verts = S.NumVertices();
  for(int i=0;i<num_verts;++i) {
    auto vS = S.vertex(i);
    auto vB0 = B0.vertex(i);
    double dx = vS[0] - vB0[0];
    double dy = vS[1] - vB0[1];
    double dz = vS[2] - vB0[2];

    CostFunction *costfun = new AutoDiffCostFunction<PointResidual, 3, 46>(
      new PointResidual(dx, dy, dz, i, dB));
    problem.AddResidualBlock(costfun, NULL, w.data());
  }

  // add prior if necessary
  if( fabs(w_prior) > 1e-6 ) {
    problem.AddResidualBlock(
      new AutoDiffCostFunction<PriorResidue, 46, 46>(new PriorResidue(wp.data())),
      NULL, w.data());
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


vector<BasicMesh> BlendshapeRefiner::RefineBlendshapes(const vector <BasicMesh> &S,
                                                       const vector <Array2D<double>> &Sgrad,
                                                       const vector <BasicMesh> &B, const BasicMesh &B00,
                                                       const vector <VectorXd> &alpha, double beta, double gamma,
                                                       const Array2D<double> prior, const Array2D<double> w_prior,
                                                       const vector<int> stationary_indices) {
  int nfaces = B00.NumFaces();
  int nverts = B00.NumVertices();

  int nposes = S.size();
  int nshapes = B.size() - 1;

  // compute the deformation gradients for B00
  Array2D<double> B00grad(nfaces, 9);
  Array1D<double> DB0(nfaces);
  for(int i=0;i<nfaces;++i) {
    auto GD = triangleGradient2(B00, i);
    auto B00i_ptr = B00grad.rowptr(i);
    for(int k=0;k<9;++k) B00i_ptr[k] = GD.first(k);
    DB0(i) = GD.second;
  }

  const int nrows_data = 9 * nposes;
  const int ncols_data = 9 * (nshapes + 1);

  const double w_data = 10.0;
  MatrixXd A_data = MatrixXd::Zero(nrows_data, ncols_data);
  for(int i=0;i<nposes;++i) {
    cout << "alpha[" << i << "] = " << alpha[i].transpose() << endl;
    A_data.block(i*9, 0, 9, 9) = MatrixXd::Identity(9, 9);
    for(int j=1;j<=nshapes;++j) {
      A_data.block(i*9, j*9, 9, 9) = alpha[i](j) * MatrixXd::Identity(9, 9);
    }
  }
  A_data *= w_data;

  ColorStream(ColorOutput::Blue) << "computing per-face gradients.";
  vector<MatrixXd> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    // b_data
    VectorXd b_data(nrows_data);
    for(int i=0;i<nposes;++i) {
      auto Sgrad_j = Sgrad[i];
      // the pointer to the j-th face
      auto Sgrad_ij = Sgrad_j.rowptr(j);
      for(int k=0;k<9;++k) b_data(i*9+k) = Sgrad_ij[k];
    }
    b_data *= w_data;

    // A_reg
    const int nrows_reg = 9 * nshapes;
    const int ncols_reg = 9 * (nshapes + 1);
    const double w_reg = 0.025;
    MatrixXd A_reg = MatrixXd::Zero(nrows_reg, ncols_reg);
    for(int i=0;i<nposes;++i) {
      MatrixXd A_reg_i = MatrixXd::Zero(9, ncols_reg);

      const double wij = beta * w_prior(i, j);

      const double* g = prior.rowptr(i) + j * 9;
      for(int ii=0;ii<3;++ii) {
        for(int jj=0;jj<3;++jj) {
          A_reg_i.block(ii*3, jj*3, 3, 3) = -1 * MatrixXd::Identity(3, 3) * g[ii*3+jj] * wij;
        }
      }

      A_reg_i.block(0, 0, 9, 9) += MatrixXd::Identity(9, 9) * wij;
      A_reg_i.block(0, (i+1)*9, 9, 9) = MatrixXd::Identity(9, 9) * wij;

      A_reg.middleRows(i*9, 9) = A_reg_i;
    }
    A_reg *= w_reg;

    // b_reg
    VectorXd b_reg = VectorXd::Zero(nrows_reg);
    b_reg *= w_reg;

    // A_sim
    const double w_sim = 0.1;
    const int nrows_sim = 9;
    const int ncols_sim = 9 * (nshapes + 1);
    MatrixXd A_sim = MatrixXd::Zero(nrows_sim, ncols_sim);
    A_sim.block(0, 0, 9, 9) = MatrixXd::Identity(9, 9) * w_sim;

    // b_sim
    VectorXd b_sim(nrows_sim);
    for(int k=0;k<9;++k) b_sim(k) = B00grad(j, k) * w_sim;

    const int nrows_total = nrows_data + nrows_reg + nrows_sim;
    const int ncols_total = ncols_data;
    MatrixXd A(nrows_total, ncols_total);
    A.topRows(nrows_data) = A_data;
    A.middleRows(nrows_data, nrows_reg) = A_reg;
    A.bottomRows(nrows_sim) = A_sim;

    VectorXd b(nrows_total);
    b.topRows(nrows_data) = b_data;
    b.middleRows(nrows_data, nrows_reg) = b_reg;
    b.bottomRows(nrows_sim) = b_sim;

    VectorXd Mv = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Store it into M
    M[j] = MatrixXd(nshapes + 1, 9);
    for(int i=0;i<=nshapes;++i) {
      M[j].row(i) = Mv.middleRows(i*9, 9).transpose();
    }
  }
  ColorStream(ColorOutput::Blue) << "done.";

  // reconstruct the blendshapes now
  ColorStream(ColorOutput::Blue) << "reconstructing blendshapes.";
  MeshTransferer transferer;
  transferer.setSource(B00);
  transferer.setTarget(B00);
  transferer.setStationaryVertices(stationary_indices);

  vector<BasicMesh> B_new(nshapes + 1);

  // recovery the neutral shape
  vector<PhGUtils::Matrix3x3d> Bgrad_0(nfaces);
  for(int j=0;j<nfaces;++j) {
    auto &Mj = M[j];
    PhGUtils::Matrix3x3d M00j(B00grad.rowptr(j), false);
    PhGUtils::Matrix3x3d M0j;
    for(int k=0;k<9;++k) M0j(k) = Mj(0, k);
    PhGUtils::Matrix3x3d Bgrad_0j = (M0j * M00j.inv()).transposed();
    for(int k=0;k<9;++k) Bgrad_0[j](k) = Bgrad_0j(k);
  }
  B_new[0] = transferer.transfer(Bgrad_0);
  auto& B0 = B_new[0];

  // compute deformation gradients
  Array2D<double> B0grad(nfaces, 9);
  Array1D<double> DB(nfaces);
  for(int i=0;i<nfaces;++i) {
    auto GD = triangleGradient2(B0, i);
    auto B0i_ptr = B0grad.rowptr(i);
    for(int k=0;k<9;++k) B0i_ptr[k] = GD.first(k);
    DB(i) = GD.second;
  }

  transferer.setSource(B0);
  transferer.setTarget(B0);
  transferer.setStationaryVertices(stationary_indices);

  // recovery all other shapes
  for(int i=1;i<=nshapes;++i) {
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

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B_new[i].Write("B_refined_"+to_string(i)+".obj");
  }

  return B_new;
}

namespace utils {
  void pause() {
    std::cin.ignore(std::numeric_limits<int>::max(), '\n');
    std::cout << "Press enter to continue...";
    std::cin.get();
  }
}

void BlendshapeRefiner::Refine() {
  // [Step 1]: deform the inintial training shapes with the input point clouds
  CreateTrainingShapes();

  // [Step 2]: create a set of initial blendshapes using initial neutral face mesh and template blendshapes
  InitializeBlendshapes();

  // [Step 3]: blendshape refinement

  // [blendshape refinement] data preparation
  auto& A0 = A[0];
  const int num_faces = A0.NumFaces();
  Array2D<double> MA0 = Array2D<double>::zeros(num_faces, 9);
  for(int j=0;j<num_faces;++j) {
    auto MA0j = triangleGradient(A0, j);
    auto MA0j_ptr = MA0.rowptr(j);
    for(int k=0;k<9;++k) MA0j_ptr[k] = MA0j(k);
  }

  const double kappa = 0.1;
  Array2D<double> prior = Array2D<double>::zeros(num_shapes, 9*num_faces);
  Array2D<double> w_prior = Array2D<double>::zeros(num_shapes, num_faces);
  for(int i=0;i<num_shapes;++i) {
    // prior for shape i
    auto &Ai = A[i+1];
    Array2D<double> Pi = Array2D<double>::zeros(num_faces, 9);
    auto w_prior_i = w_prior.rowptr(i);
    for(int j=0;j<num_faces;++j) {
      auto MAij = triangleGradient(Ai, j);
      auto MA0j_ptr = MA0.rowptr(j);
      // create a 3x3 matrix from MA0j_ptr
      auto MA0j = PhGUtils::Matrix3x3d(MA0j_ptr, false);
      auto GA0Ai = MAij * (MA0j.inv());
      auto Pij = GA0Ai;
      double MAij_norm = (MAij-MA0j).norm();
      w_prior_i[j] = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0);

      auto Pij_ptr = Pi.rowptr(j);
      for(int k=0;k<9;++k) Pij_ptr[k] = Pij(k);
    }
    auto prior_i = prior.rowptr(i);
    // prior(i,:) = Pi;
    memcpy(prior_i, Pi.data.get(), sizeof(double)*num_faces*9);
  }

  ofstream fprior("prior.txt");
  fprior<<prior;
  fprior.close();

  ofstream fwprior("w_prior.txt");
  fwprior<<w_prior;
  fwprior.close();

  ColorStream(ColorOutput::Blue)<< "priors computed.";

  // compute the delta shapes
  vector<MatrixX3d> dB(num_shapes);
  for(int i=0;i<num_shapes;++i) {
    dB[i] = B[i+1].vertices() - B[0].vertices();
  }

  // make a copy of B0 for regularization
  BasicMesh B00 = B[0];
  vector<int> stationary_indices = B00.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  });

  // estimate initial set of blendshape weights
  vector<VectorXd> alpha(num_poses);
  for(int i=0;i<num_poses;++i) {
    alpha[i] = image_bundles[i].params.params_model.Wexp_FACS;
  }
  auto alpha_init = alpha;

  // compute deformation gradients for S
  // Note: this deformation gradient cannot be used to call
  // MeshTransferer::transfer directly. See this function for
  // details
  vector<Array2D<double>> Sgrad(num_poses);
  for(int i=0;i<num_poses;++i) {
    Sgrad[i] = Array2D<double>::zeros(num_faces, 9);
    for(int j=0;j<num_faces;++j) {
      // assign the reshaped gradient to the j-th row of Sgrad[i]
      auto Sij = triangleGradient(S[i], j);
      auto Sij_ptr = Sgrad[i].rowptr(j);
      for(int k=0;k<9;++k) Sij_ptr[k] = Sij(k);
    }
  }

  ColorStream(ColorOutput::Red)<< "initialization done.";

  // [blendshape refinement] Main loop
  bool converged = false;
  double ALPHA_THRES = 1e-6;
  double B_THRES = 1e-6;
  double beta_max = 0.05, beta_min = 0.01;
  double gamma_max = 0.01, gamma_min = 0.01;
  double eta_max = 1.0, eta_min = 0.1;
  int iters = 0;
  const int maxIters = 1;
  MatrixXd B_error = MatrixXd::Zero(maxIters, num_shapes + 1);
  MatrixXd S_error = MatrixXd::Zero(maxIters, num_poses);
  while( !converged && iters < maxIters ){
    cout << "iteration " << iters << " ..." << endl;
    converged = true;
    ++iters;

    // [Step a]: Refine blendshapes
    double beta = sqrt(iters/maxIters) * (beta_min - beta_max) + beta_max;
    double gamma = gamma_max + iters/maxIters*(gamma_min-gamma_max);
    double eta = eta_max + iters/maxIters*(eta_min-eta_max);

    auto B_new = RefineBlendshapes(S, Sgrad, B, B00, alpha, beta, gamma, prior, w_prior, stationary_indices);

    // convergence test
    VectorXd B_norm(num_shapes+1);
    for(int i=0;i<=num_shapes;++i) {
      auto Bdiff = B[i].vertices() - B_new[i].vertices();
      B_norm(i) = Bdiff.norm();
      B[i].vertices() = B_new[i].vertices();
      //B{i+1} = alignMesh(B{i+1}, B{1}, stationary_indices);
    }
    ColorStream(ColorOutput::Green)<< "Bnorm = " << B_norm.transpose();
    converged = converged & (B_norm.maxCoeff() < B_THRES);

    // [Step b]: Update expression weights

    // update delta shapes
    for(int i=0;i<num_shapes;++i) {
      dB[i] = B[i+1].vertices() - B[0].vertices();
    }

    // update weights
    VectorXd alpha_norm(num_poses);
    vector<VectorXd> alpha_new(num_poses);
    for(int i=0;i<num_poses;++i) {
      alpha_new[i] = EstimateWeights(S[i], B[0], dB, alpha[i], alpha_init[i], eta, 2);
      auto alpha_diff = alpha_new[i] - alpha[i];
      alpha_norm(i) = alpha_diff.norm();
    }
    alpha = alpha_new;
    ColorStream(ColorOutput::Green)<< "norm(alpha) = " << alpha_norm.transpose();
    converged = converged & (alpha_norm.maxCoeff() < ALPHA_THRES);

    for(int i=0;i<num_poses;++i) {
      // make a copy of B0
      auto Ti = B[0];
      for(int j=0;j<num_shapes;++j) {
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
    for(int i=0;i<num_poses;++i) {
      MeshDeformer deformer;
      deformer.setSource(S[i]);
      S[i] = deformer.deformWithPoints(point_clouds[i], PointCloud(), 5);
    }

    // compute deformation gradients for S
    for(int i=0;i<num_poses;++i) {
      Sgrad[i] = Array2D<double>::zeros(num_faces, 9);
      for(int j=0;j<num_faces;++j) {
        // assign the reshaped gradient to the j-th row of Sgrad[i]
        auto Sij = triangleGradient(S[i], j);
        auto Sij_ptr = Sgrad[i].rowptr(j);
        for(int k=0;k<9;++k) Sij_ptr[k] = Sij(k);
      }
    }
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B[i].Write("B_"+to_string(i)+".obj");
  }

  // write out the blendshapes
  for(int i=0;i<num_poses;++i) {
    S[i].Write("S_"+to_string(i)+".obj");
  }
}
