#include "blendshaperefiner.h"
#include "cereswrapper.h"
#include "meshdeformer.h"
#include "meshtransferer.h"

#include "MultilinearReconstruction/costfunctions.h"

namespace utils {
  void pause() {
    std::cout << "Press enter to continue...";
    std::cin.ignore(std::numeric_limits<int>::max(), '\n');
  }
}

BlendshapeRefiner::BlendshapeRefiner(json settings) {
  if(!settings.empty()) {
    use_init_blendshapes = settings["use_init_blendshapes"];
    do_subdivision = settings["subdivision"];
  } else {
    use_init_blendshapes = false;
    do_subdivision = false;
  }

  model = MultilinearModel("/home/phg/Data/Multilinear/blendshape_core.tensor");

  model_prior.load("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");

  template_mesh = BasicMesh("/home/phg/Data/Multilinear/template.obj");
  cout << "Blendshape refiner created." << endl;
}

void BlendshapeRefiner::SetResourcesPath(const string& path) {
  cout << "Setting resources path to " << path << endl;
  resources_path = fs::path(path);
  cout << "done." << endl;
}

void BlendshapeRefiner::SetReconstructionsPath(const string& path) {
  cout << "Setting reconstructions path to " << path << endl;
  reconstructions_path = fs::path(path);
  cout << "done." << endl;
}

void BlendshapeRefiner::SetPointCloudsPath(const string& path) {
  cout << "Setting point clouds path to " << path << endl;
  point_clouds_path = fs::path(path);
  cout << "done." << endl;
}

void BlendshapeRefiner::SetInputBlendshapesPath(const string& path) {
  cout << "Setting input blendshapes path to " << path << endl;
  input_blendshapes_path = fs::path(path);
  cout << "done." << endl;
}

void BlendshapeRefiner::SetBlendshapesPath(const string& path) {
  cout << "Setting blendshapes path to " << path << endl;
  blendshapes_path = fs::path(path);
  if(!fs::exists(blendshapes_path)) {
    try{
      cout << "Creating blendshapes directory " << blendshapes_path.string() << endl;
      fs::create_directory(blendshapes_path);
    } catch(exception& e) {
      cout << e.what() << endl;
      exit(1);
    }
  }
}

void BlendshapeRefiner::LoadTemplateMeshes(const string &path, const string &basename) {
  A.resize(num_shapes + 1);

  for(int i=0;i<=num_shapes;++i) {
    A[i].LoadOBJMesh(path + basename + to_string(i) + ".obj");
  }
}

void BlendshapeRefiner::LoadInitialBlendshapes() {
  Binit.resize(num_shapes + 1);

  for(int i=0;i<=num_shapes;++i) {
    Binit[i].LoadOBJMesh( (input_blendshapes_path / fs::path("B_" + to_string(i) + ".obj")).string() );
  }
}

void BlendshapeRefiner::LoadSelectionFile(const string& selection_filename) {
  selection_indices = LoadIndices((resources_path / fs::path(selection_filename)).string());

  for(auto idx : selection_indices) {
    selection_to_order_map[idx] = -1;
  }
}

namespace {
  int getIndex(const string& filename) {
    istringstream iss(filename);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, '.')) {
        if (!token.empty())
            tokens.push_back(token);
    }
    return std::stoi(tokens[0]);
  }
}

void BlendshapeRefiner::LoadInputReconstructionResults(const string &settings_filename) {
  vector<pair<string, string>> image_points_filenames = ParseSettingsFile( (resources_path / fs::path(settings_filename)).string() );

  image_bundles.resize(selection_indices.size());
  int order = 0;
  for(auto& p : image_points_filenames) {
    int img_idx = getIndex(p.first);
    if( selection_to_order_map.find(img_idx) == selection_to_order_map.end() ) {
      ++order;
      continue;
    }

    int selector_order = -1;
    for(int j=0;j<selection_indices.size();++j) {
      if(selection_indices[j] == img_idx) {
        selector_order = j;
        break;
      }
    }

    cout << p.first << endl;
    selection_to_order_map[img_idx] = order++;

    fs::path image_filename = resources_path / fs::path(p.first);
    fs::path pts_filename = resources_path / fs::path(p.second);
    fs::path res_filename = reconstructions_path / fs::path(p.first + ".res");
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string());
    auto recon_results = LoadReconstructionResult(res_filename.string());
    image_bundles[selector_order] = (ImageBundle(image_points_pair.first, image_points_pair.second, recon_results));
  }
  //utils::pause();

  // Set the number of poses
  num_poses = selection_indices.size();
}

MatrixXd BlendshapeRefiner::LoadPointCloud(const string &filename, const glm::dmat4& Rmat_inv) {
  ifstream fin(filename);
  vector<Vector3d> points;
  points.reserve(100000);
  while(fin) {
    double x, y, z;
    fin >> x >> y >> z;

    // rotate the input point cloud to regular view
    glm::dvec4 pt0 =  Rmat_inv * glm::dvec4(x, y, z, 1.0);

    points.push_back(Vector3d(pt0.x, pt0.y, pt0.z));
  }

  MatrixXd P(points.size(), 3);
  for(int i=0;i<points.size();++i) {
    P.row(i) = points[i];
  }

  return P;
}

void BlendshapeRefiner::LoadInputPointClouds() {
  // Load all points files
  point_clouds.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    int img_idx = selection_indices[i];
    int point_cloud_idx = selection_to_order_map[img_idx];
    assert(point_cloud_idx != -1);

    bool need_rotation = true;
    glm::dmat4 R;
    if(need_rotation) {
      // Need to rotate the points back to frontal view
      glm::dmat4 Rmat_inv = glm::eulerAngleZ(-image_bundles[i].params.params_model.R[2])
                          * glm::eulerAngleX(-image_bundles[i].params.params_model.R[1])
                          * glm::eulerAngleY(-image_bundles[i].params.params_model.R[0]);
      R = Rmat_inv;
    } else {
      // just use the identity matrix
    }

    point_clouds[i] = LoadPointCloud( (point_clouds_path / fs::path("masked_optimized_point_cloud_" + to_string(point_cloud_idx) + ".txt")).string(),
                                      R );
  }
}

namespace {
  void ApplyWeights(
    BasicMesh& mesh,
    const vector<BasicMesh>& blendshapes,
    const VectorXd& weights
  ) {
    const int num_blendshapes = 46;
    MatrixX3d verts0 = blendshapes[0].vertices();
    MatrixX3d verts = verts0;
    for(int j=1;j<=num_blendshapes;++j) {
      verts += (blendshapes[j].vertices() - verts0) * weights(j);
    }
    mesh.vertices() = verts;
    mesh.ComputeNormals();
  }
}

void BlendshapeRefiner::CreateTrainingShapes() {
  S0.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    ColorStream(ColorOutput::Green)<< "creating initial training shape " << i;
    S0[i] = template_mesh;

    if(use_init_blendshapes) {
      // Synthesize from initial blendshapes
      ApplyWeights(S0[i], Binit, image_bundles[i].params.params_model.Wexp_FACS);
    } else {
      // Synthesize from multilinear model
      model.ApplyWeights(image_bundles[i].params.params_model.Wid, image_bundles[i].params.params_model.Wexp);
      S0[i].UpdateVertices(model.GetTM());
    }

    S0[i].ComputeNormals();
    S0[i].Write( InBlendshapesDirectory("Sinit_" + to_string(i) + ".obj") );
  }
  ColorStream(ColorOutput::Blue)<< "initial training shapes created.";


  auto& m = (template_mesh);
  vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
    Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
    return c[2] > -0.5;
  });

  S.resize(num_poses);
  #pragma omp parallel for
  for(int i=0;i<num_poses;++i) {
    ColorStream(ColorOutput::Green)<< "creating refined training shape " << i;
    MeshDeformer deformer;
    deformer.setSource(m);
    deformer.setValidFaces(valid_faces);
    deformer.setSource(S0[i]);

    S[i] = deformer.deformWithPoints(point_clouds[i], PointCloud(), 20);
    S[i].Write( InBlendshapesDirectory("S0_" + to_string(i) + ".obj") );
  }
  ColorStream(ColorOutput::Blue)<< "refined training shapes created.";
}

void BlendshapeRefiner::InitializeBlendshapes() {
  if(use_init_blendshapes) {
    // Nothing to do
  } else {
    // Create the initial neutral face mesh
    model.ApplyWeights(image_bundles[0].params.params_model.Wid, model_prior.Wexp0);

    Binit.resize(A.size());
    Binit[0] = template_mesh;
    Binit[0].UpdateVertices(model.GetTM());
    Binit[0].ComputeNormals();

    Binit[0].Write( InBlendshapesDirectory("Binit_0.obj") );

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
      Binit[i].Write( InBlendshapesDirectory("Binit_" + to_string(i) + ".obj") );
    }
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
    Ti.Write( InBlendshapesDirectory("S0_verify_" + std::to_string(i) + ".obj") );
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
    problem.AddResidualBlock(costfun, NULL, w.data()+1);
  }

  // add prior if necessary
  if( fabs(w_prior) > 1e-6 ) {
    problem.AddResidualBlock(
      new AutoDiffCostFunction<PriorResidue, 46, 46>(new PriorResidue(wp.data()+1)),
      NULL, w.data()+1);
  }

  /*
  ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationTerm> *reg_cost_function =
    new ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationTerm>(
      new ExpressionRegularizationTerm(10.0)
    );
  reg_cost_function->AddParameterBlock(46);
  reg_cost_function->SetNumResiduals(46);
  problem.AddResidualBlock(reg_cost_function, NULL, w.data()+1);
  */

  for(int i=0;i<46;++i) {
    problem.SetParameterLowerBound(w.data()+1, i, 0.0);
    problem.SetParameterUpperBound(w.data()+1, i, 1.0);
  }

  cout << "w0 = " << endl;
  cout << w << endl;
  // set the solver options
  Solver::Options options;

  options.max_num_iterations = 10;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;

  options.initial_trust_region_radius = 1.0;
  options.min_lm_diagonal = 1.0;
  options.max_lm_diagonal = 1.0;

  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;

  return w;
}


vector<BasicMesh> BlendshapeRefiner::RefineBlendshapes(const vector <BasicMesh> &S,
                                                       const vector <vector<PhGUtils::Matrix3x3d>> &Sgrad,
                                                       const vector <BasicMesh> &A,
                                                       const vector <BasicMesh> &B, const BasicMesh &B00,
                                                       const vector <VectorXd> &alpha,
                                                       double beta, double gamma,
                                                       const vector <vector<PhGUtils::Matrix3x3d>> &prior,
                                                       const MatrixXd& w_prior,
                                                       const vector<int> stationary_indices,
                                                       bool refine_neutral_only) {
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

  // compute the deformation gradients for B0j
  vector<Array2D<double>> B00grads(nshapes+1, Array2D<double>(nfaces, 9));
  B00grads[0] = B00grad;
  for(int j=1;j<=nshapes;++j) {
    B00grads[j].resize(nfaces, 9);
    for(int i=0;i<nfaces;++i) {
      auto GD = triangleGradient2(B[j], i);
      auto B0ji_ptr = B00grads[j].rowptr(i);
      for(int k=0;k<9;++k) B0ji_ptr[k] = GD.first(k);
    }
  }

  const int nrows_data = 9 * nposes;
  const int ncols_data = 9 * (nshapes + 1);

  using Tripletd = Eigen::Triplet<double>;
  using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  vector<Tripletd> Adata_coeffs;

  const double w_data = 1.0;
  MatrixXd A_data = MatrixXd::Zero(nrows_data, ncols_data);
  for(int i=0;i<nposes;++i) {
    int row_offset = i * 9;
    for(int k=0;k<9;++k) {
      Adata_coeffs.push_back(Tripletd(row_offset+k, k, w_data));
    }
    for(int j=1;j<=nshapes;++j) {
      int col_offset = j * 9;
      for(int k=0;k<9;++k) {
        Adata_coeffs.push_back(Tripletd(row_offset+k, col_offset+k, w_data * alpha[i](j)));
      }
    }
  }

  ColorStream(ColorOutput::Blue) << "computing per-face gradients.";
  vector<MatrixXd> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    // b_data
    VectorXd b_data(nrows_data);
    for(int i=0;i<nposes;++i) {
      auto Sgrad_ij = Sgrad[i][j];
      for(int k=0;k<9;++k) b_data(i*9+k) = Sgrad_ij(k);
    }
    b_data *= w_data;

    // A_reg
    vector<Tripletd> Areg_coeffs;
    const int nrows_reg = 9 * nshapes;
    const int ncols_reg = 9 * (nshapes + 1);

    const double w_reg = 0.5;
    //MatrixXd A_reg = MatrixXd::Zero(nrows_reg, ncols_reg);
    for(int i=0;i<nshapes;++i) {
      int row_offset = nrows_data + i * 9;
      MatrixXd A_reg_i = MatrixXd::Zero(9, ncols_reg);

      const double wij = beta * w_prior(i, j);

      const PhGUtils::Matrix3x3d& Gij = prior[i][j];

      for(int ii=0;ii<3;++ii) {
        for(int jj=0;jj<3;++jj) {
          // A_reg_i.block<3,3>(ii*3, jj*3) = -1 * MatrixXd::Identity(3, 3) * Gij(ii, jj) * wij;

          for(int k=0;k<3;++k) {
            Areg_coeffs.push_back(Tripletd(row_offset+ii*3+k, jj*3+k, -wij * Gij(ii, jj) * w_reg));
            if(ii == jj) {
              Areg_coeffs.push_back(Tripletd(row_offset+ii*3+k, jj*3+k, wij* w_reg));
            }
          }
        }
      }

      //A_reg_i.block<9,9>(0, 0) += MatrixXd::Identity(9, 9) * wij;
      //A_reg_i.block<9,9>(0, (i+1)*9) = MatrixXd::Identity(9, 9) * wij;
      int col_offset = (i+1) * 9;
      for(int k=0;k<9;++k) {
        Areg_coeffs.push_back(Tripletd(row_offset + k, col_offset+k, wij*w_reg));
      }

      //A_reg.middleRows(i*9, 9) = A_reg_i;
    }
    //A_reg *= w_reg;

    // b_reg
    VectorXd b_reg = VectorXd::Zero(nrows_reg);
    b_reg *= w_reg;

    // A_sim
    // FIXME try include all shapes to provide better constraints
    double w_sim;
    int nrows_sim;
    int ncols_sim;
    vector<Tripletd> Asim_coeffs;
    VectorXd b_sim;
    if(refine_neutral_only) {
      w_sim = 0.01;
      nrows_sim = 9;
      ncols_sim = 9 * (nshapes + 1);

      for(int k=0;k<9;++k) {
        Asim_coeffs.push_back(Tripletd(nrows_data + nrows_reg + k, k, w_sim));
      }

      // b_sim
      b_sim = VectorXd(nrows_sim);
      for(int k=0;k<9;++k) b_sim(k) = B00grad(j, k) * w_sim;
    } else {
      w_sim = 0.25;
      nrows_sim = 9 * (nshapes + 1);
      ncols_sim = 9 * (nshapes + 1);

      Asim_coeffs;
      for(int k=0;k<9*(nshapes+1);++k) {
        Asim_coeffs.push_back(Tripletd(nrows_data + nrows_reg + k, k, w_sim));
      }

      // b_sim
      b_sim = VectorXd(nrows_sim);
      for(int k=0;k<9;++k) b_sim(k) = B00grads[0](j, k) * w_sim;
      for(int i=1;i<=nshapes;++i) {
        for(int k=0;k<9;++k) b_sim(i*9+k) = (B00grads[i](j, k) - B00grads[0](j, k))* w_sim;
      }
    }

    const int nrows_total = nrows_data + nrows_reg + nrows_sim;
    const int ncols_total = ncols_data;


    //MatrixXd A(nrows_total, ncols_total);
    //A.topRows(nrows_data) = A_data;
    //A.middleRows(nrows_data, nrows_reg) = A_reg;
    //A.bottomRows(nrows_sim) = A_sim;

    Eigen::SparseMatrix<double> A(nrows_total, ncols_total);
    vector<Tripletd> A_coeffs = Adata_coeffs;
    A_coeffs.insert(A_coeffs.end(), Areg_coeffs.begin(), Areg_coeffs.end());
    A_coeffs.insert(A_coeffs.end(), Asim_coeffs.begin(), Asim_coeffs.end());

    if(0) {
      ofstream fA("A_coeffs.txt");
      for(Tripletd& coeffs : A_coeffs) {
        fA << coeffs.row() << ',' << coeffs.col() << ',' << coeffs.value() << endl;
      }
      fA.close();
    }

    A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
    A.makeCompressed();

    Eigen::SparseMatrix<double> AtA = (A.transpose() * A).pruned();

    const double epsilon = 0.0;//1e-9;
    Eigen::SparseMatrix<double> eye(ncols_total, ncols_total);
    for(int j=0;j<ncols_total;++j) eye.insert(j, j) = epsilon;
    AtA += eye;

    CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);

    VectorXd b(nrows_total);
    b.topRows(nrows_data) = b_data;
    b.middleRows(nrows_data, nrows_reg) = b_reg;
    b.bottomRows(nrows_sim) = b_sim;

    /*
    JacobiSVD<MatrixXd> svd(A.transpose() * A);
    double cond = svd.singularValues()(0)
                  / svd.singularValues()(svd.singularValues().size()-1);
    cout << cond << endl;

    VectorXd Mv = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    */

    VectorXd Atb = A.transpose() * b;
    VectorXd Mv = solver.solve(Atb);
    if(solver.info()!=Success) {
      cerr << "Failed to solve A\\b." << endl;
      exit(-1);
    }

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

  // @FIXME try do deformation transfer on the new neutral shape instead of using the
  // computed triangle gradients.

  // recovery all other shapes
  for(int i=1;i<=nshapes;++i) {
    ColorStream(ColorOutput::Green)<< "reconstructing blendshapes " << i;
    if(refine_neutral_only) {
      transferer.setSource(A[0]);
      B_new[i] = transferer.transfer(A[i]);
    } else {
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
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B_new[i].Write( InBlendshapesDirectory("B_refined_"+to_string(i)+".obj") );
  }

  return B_new;
}

void BlendshapeRefiner::Refine() {
  // [Step 0]: load initial blendshapes if possible
  if(use_init_blendshapes) LoadInitialBlendshapes();

  // [Step 1]: deform the inintial training shapes with the input point clouds
  CreateTrainingShapes();

  // [Step 2]: create a set of initial blendshapes using initial neutral face mesh and template blendshapes
  InitializeBlendshapes();

  // [Step 3]: blendshape refinement

  // [blendshape refinement] data preparation
  auto& A0 = A[0];
  int num_faces = A0.NumFaces();

  auto ComputerPrior = [](const vector<BasicMesh>& A) {
    ColorStream(ColorOutput::Blue) << "Computing prior ...";
    auto& A0 = A[0];
    int num_faces = A0.NumFaces();
    int num_shapes = A.size()-1;
    const double kappa = 0.1;

    vector<PhGUtils::Matrix3x3d> MA0(num_faces);
    for(int j=0;j<num_faces;++j) {
      MA0[j] = triangleGradient(A0, j);
    }

    vector<vector<PhGUtils::Matrix3x3d>> prior(num_shapes, vector<PhGUtils::Matrix3x3d>(num_faces));
    MatrixXd w_prior(num_shapes, num_faces);
    for(int i=0;i<num_shapes;++i) {
      // prior for shape i
      auto &Ai = A[i+1];
      for(int j=0;j<num_faces;++j) {
        auto MAij = triangleGradient(Ai, j);
        auto GA0Ai = MAij * (MA0[j].inv());
        prior[i][j] = GA0Ai;
        double MAij_norm = (MAij-MA0[j]).norm();
        w_prior(i, j) = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0);
      }
    }

    return make_pair(prior, w_prior);
  };

  // Computer prior from A
  vector<vector<PhGUtils::Matrix3x3d>> prior;
  MatrixXd w_prior;
  tie(prior, w_prior) = ComputerPrior(A);

  /*
  ofstream fprior("prior.txt");
  fprior<<prior;
  fprior.close();

  ofstream fwprior("w_prior.txt");
  fwprior<<w_prior;
  fwprior.close();
  */

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
  vector<vector<PhGUtils::Matrix3x3d>> Sgrad(num_poses, vector<PhGUtils::Matrix3x3d>(num_faces));
  for(int i=0;i<num_poses;++i) {
    for(int j=0;j<num_faces;++j) {
      // assign the reshaped gradient to the j-th row of Sgrad[i]
      Sgrad[i][j] = triangleGradient(S[i], j);
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
  const int maxIters = 2;   // This will do 2 subdivisions
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

    // Refine the neutral shape only
    auto B_new = RefineBlendshapes(S, Sgrad, A, B, B00, alpha, beta, gamma, prior, w_prior, stationary_indices);

    // Refine all blendshapes
    B_new = RefineBlendshapes(S, Sgrad, A, B_new, B_new[0], alpha, beta, gamma, prior, w_prior, stationary_indices, false);

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
        Ti.vertices() += alpha[i](j+1) * dB[j];
      }
      //Ti = alignMesh(Ti, S0{i});
      //S_error(iters, i) = sqrt(max(sum((Ti.vertices-S0{i}.vertices).^2, 2)));

      // update the reconstructed mesh
      S[i] = Ti;

      S[i].Write( InBlendshapesDirectory("S_fitted_" + to_string(i) + ".obj") );
    }

    if(iters == maxIters) break;

    // Optional: subdivide all meshes
    if(do_subdivision){
      ColorStream(ColorOutput::Blue) << "Subdividing the meshes...";
      // Subdivide every mesh
      // Subdivide A and update prior, w_prior, stationary_indices
      ColorStream(ColorOutput::Blue) << "Subdividing the template meshes...";
      for(auto &Ai : A) {
        Ai.BuildHalfEdgeMesh();
        Ai.Subdivide();
      }
      tie(prior, w_prior) = ComputerPrior(A);

      // Subdivide B and B00
      ColorStream(ColorOutput::Blue) << "Subdividing the blendshape meshes...";
      for(auto &Bi : B) {
        Bi.BuildHalfEdgeMesh();
        Bi.Subdivide();
      }
      stationary_indices = B00.filterVertices([=](const Vector3d& v) {
        return v[2] <= -0.45;
      });
      B00.BuildHalfEdgeMesh();
      B00.Subdivide();

      // Subdivide S, no need to update Sgrad because S will be deformed later
      ColorStream(ColorOutput::Blue) << "Subdividing the training meshes...";
      for(auto &Si : S) {
        Si.BuildHalfEdgeMesh();
        Si.Subdivide();
      }

      num_faces *= 4;
    }

    // The reconstructed mesh are updated using the new set of blendshape
    // weights, need to use laplacian deformation to refine them again
    ColorStream(ColorOutput::Blue) << "Updating the training shapes...";
    #pragma omp parallel for
    for(int i=0;i<num_poses;++i) {
      ColorStream(ColorOutput::Green) << "Updating training shape " << i << "...";
      auto& m = B00;
      MeshDeformer deformer;
      deformer.setSource(S[i]);

      vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
        Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
        return c[2] > -0.5;
      });
      deformer.setValidFaces(valid_faces);

      S[i] = deformer.deformWithPoints(point_clouds[i], PointCloud(), 5);
      S[i].Write( InBlendshapesDirectory("S_refined_" + to_string(i) + ".obj") );
    }

    // compute deformation gradients for S
    for(int i=0;i<num_poses;++i) {
      Sgrad[i].resize(num_faces);
      for(int j=0;j<num_faces;++j) {
        // assign the reshaped gradient to the j-th row of Sgrad[i]
        Sgrad[i][j] = triangleGradient(S[i], j);
      }
    }
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B[i].Write( InBlendshapesDirectory("B_"+to_string(i)+".obj") );
  }

  // write out the blendshapes
  for(int i=0;i<num_poses;++i) {
    S[i].Write( InBlendshapesDirectory("S_"+to_string(i)+".obj") );
  }
}

vector<BasicMesh> BlendshapeRefiner::RefineBlendshapes_EBFR(const vector <BasicMesh> &S,
                                                       const vector <vector<PhGUtils::Matrix3x3d>> &Sgrad,
                                                       const vector <BasicMesh> &A,
                                                       const vector <BasicMesh> &B, const BasicMesh &B00,
                                                       const vector <VectorXd> &alpha,
                                                       double beta, double gamma,
                                                       const vector <vector<PhGUtils::Matrix3x3d>> &prior,
                                                       const MatrixXd& w_prior,
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
  const int ncols_data = 9 * nshapes;

  using Tripletd = Eigen::Triplet<double>;
  using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  vector<Tripletd> Adata_coeffs;

  const double w_data = 1.0;
  for(int i=0;i<nposes;++i) {
    int row_offset = i * 9;
    for(int j=0;j<nshapes;++j) {
      int col_offset = j * 9;
      for(int k=0;k<9;++k) {
        Adata_coeffs.push_back(Tripletd(row_offset+k, col_offset+k, w_data * alpha[i](j+1)));
      }
    }
  }

  ColorStream(ColorOutput::Blue) << "computing per-face gradients.";
  vector<MatrixXd> M(nfaces);

#pragma omp parallel for
  for(int j=0;j<nfaces;++j) {
    // b_data
    VectorXd b_data(nrows_data);
    for(int i=0;i<nposes;++i) {
      auto Sgrad_ij = Sgrad[i][j];
      for(int k=0;k<9;++k) b_data(i*9+k) = Sgrad_ij(k) - B00grad.rowptr(j)[k];
    }
    b_data *= w_data;

    // A_reg
    vector<Tripletd> Areg_coeffs;
    const int nrows_reg = 9 * nshapes;
    const int ncols_reg = 9 * nshapes;

    const double w_reg = 1.0;
    //MatrixXd A_reg = MatrixXd::Zero(nrows_reg, ncols_reg);
    for(int i=0;i<nshapes;++i) {
      int row_offset = nrows_data + i * 9;
      int col_offset = i * 9;

      const double wij = beta * w_prior(i, j);

      for(int k=0;k<9;++k) {
        Areg_coeffs.push_back(Tripletd(row_offset + k, col_offset+k, wij*w_reg));
      }
    }

    // b_reg
    VectorXd b_reg = VectorXd::Zero(nrows_reg);
    auto MB0j = PhGUtils::Matrix3x3d(B00grad.rowptr(j), false);;
    for(int i=0;i<nshapes;++i) {
      const double wij = beta * w_prior(i, j);
      auto Pij = prior[i][j];
      Pij = Pij * MB0j - MB0j;
      for(int k=0;k<9;++k) b_reg(i*9+k) = wij * Pij(k);
    }

    const int nrows_total = nrows_data + nrows_reg;
    const int ncols_total = ncols_data;


    //MatrixXd A(nrows_total, ncols_total);
    //A.topRows(nrows_data) = A_data;
    //A.middleRows(nrows_data, nrows_reg) = A_reg;
    //A.bottomRows(nrows_sim) = A_sim;

    Eigen::SparseMatrix<double> A(nrows_total, ncols_total);
    vector<Tripletd> A_coeffs = Adata_coeffs;
    A_coeffs.insert(A_coeffs.end(), Areg_coeffs.begin(), Areg_coeffs.end());

    if(0) {
      ofstream fA("A_coeffs.txt");
      for(Tripletd& coeffs : A_coeffs) {
        fA << coeffs.row() << ',' << coeffs.col() << ',' << coeffs.value() << endl;
      }
      fA.close();
    }

    A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
    A.makeCompressed();

    Eigen::SparseMatrix<double> AtA = (A.transpose() * A).pruned();

    const double epsilon = 1e-6;
    Eigen::SparseMatrix<double> eye(ncols_total, ncols_total);
    for(int j=0;j<ncols_total;++j) eye.insert(j, j) = epsilon;
    AtA += eye;

    CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);

    VectorXd b(nrows_total);
    b.topRows(nrows_data) = b_data;
    b.bottomRows(nrows_reg) = b_reg;

    VectorXd Atb = A.transpose() * b;
    VectorXd Mv = solver.solve(Atb);
    if(solver.info()!=Success) {
      cerr << "Failed to solve A\\b." << endl;
      exit(-1);
    }

    // Store it into M
    M[j] = MatrixXd(nshapes, 9);
    for(int i=0;i<nshapes;++i) {
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

  B_new[0] = B00;
  auto& B0 = B_new[0];

  // recovery all other shapes
  for(int i=1;i<=nshapes;++i) {
    ColorStream(ColorOutput::Green)<< "reconstructing blendshapes " << i;
    vector<PhGUtils::Matrix3x3d> Bgrad_i(nfaces);
    for(int j=0;j<nfaces;++j) {
      auto &Mj = M[j];
      PhGUtils::Matrix3x3d M0j(B00grad.rowptr(j), false);

      PhGUtils::Matrix3x3d Mij;
      for(int k=0;k<9;++k) Mij(k) = Mj(i-1, k);

      PhGUtils::Matrix3x3d Bgrad_ij = ((Mij + M0j) * M0j.inv()).transposed();
      for(int k=0;k<9;++k) Bgrad_i[j](k) = Bgrad_ij(k);
    }
    B_new[i] = transferer.transfer(Bgrad_i);
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B_new[i].Write( InBlendshapesDirectory("B_refined_"+to_string(i)+".obj") );
  }

  return B_new;
}

void BlendshapeRefiner::Refine_EBFR() {
  // [Step 1]: deform the inintial training shapes with the input point clouds
  CreateTrainingShapes();

  // [Step 2]: create a set of initial blendshapes using initial neutral face mesh and template blendshapes
  InitializeBlendshapes();

  // [Step 3]: blendshape refinement

  // [blendshape refinement] data preparation
  auto& A0 = A[0];
  int num_faces = A0.NumFaces();

  auto ComputerPrior = [](const vector<BasicMesh>& A) {
    ColorStream(ColorOutput::Blue) << "Computing prior ...";
    auto& A0 = A[0];
    int num_faces = A0.NumFaces();
    int num_shapes = A.size()-1;
    const double kappa = 0.1;

    vector<PhGUtils::Matrix3x3d> MA0(num_faces);
    for(int j=0;j<num_faces;++j) {
      MA0[j] = triangleGradient(A0, j);
    }

    vector<vector<PhGUtils::Matrix3x3d>> prior(num_shapes, vector<PhGUtils::Matrix3x3d>(num_faces));
    MatrixXd w_prior(num_shapes, num_faces);
    for(int i=0;i<num_shapes;++i) {
      // prior for shape i
      auto &Ai = A[i+1];
      for(int j=0;j<num_faces;++j) {
        auto MAij = triangleGradient(Ai, j);
        auto GA0Ai = MAij * (MA0[j].inv());
        prior[i][j] = GA0Ai;
        double MAij_norm = (MAij-MA0[j]).norm();
        w_prior(i, j) = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0);
      }
    }

    return make_pair(prior, w_prior);
  };

  // Computer prior from A
  vector<vector<PhGUtils::Matrix3x3d>> prior;
  MatrixXd w_prior;
  tie(prior, w_prior) = ComputerPrior(A);

  /*
  ofstream fprior("prior.txt");
  fprior<<prior;
  fprior.close();

  ofstream fwprior("w_prior.txt");
  fwprior<<w_prior;
  fwprior.close();
  */

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
  vector<vector<PhGUtils::Matrix3x3d>> Sgrad(num_poses, vector<PhGUtils::Matrix3x3d>(num_faces));
  for(int i=0;i<num_poses;++i) {
    for(int j=0;j<num_faces;++j) {
      // assign the reshaped gradient to the j-th row of Sgrad[i]
      Sgrad[i][j] = triangleGradient(S[i], j);
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
  const int maxIters = 2;   // This will do 2 subdivisions
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

    // Refine the neutral shape only
    auto B_new = RefineBlendshapes_EBFR(S, Sgrad, A, B, B00, alpha, beta, gamma, prior, w_prior, stationary_indices);

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
        Ti.vertices() += alpha[i](j+1) * dB[j];
      }
      //Ti = alignMesh(Ti, S0{i});
      //S_error(iters, i) = sqrt(max(sum((Ti.vertices-S0{i}.vertices).^2, 2)));

      // update the reconstructed mesh
      S[i] = Ti;

      S[i].Write( InBlendshapesDirectory("S_fitted_" + to_string(i) + ".obj") );
    }

    if(iters == maxIters) break;

    // Optional: subdivide all meshes
    if(do_subdivision){
      ColorStream(ColorOutput::Blue) << "Subdividing the meshes...";
      // Subdivide every mesh
      // Subdivide A and update prior, w_prior, stationary_indices
      ColorStream(ColorOutput::Blue) << "Subdividing the template meshes...";
      for(auto &Ai : A) {
        Ai.BuildHalfEdgeMesh();
        Ai.Subdivide();
      }
      tie(prior, w_prior) = ComputerPrior(A);

      // Subdivide B and B00
      ColorStream(ColorOutput::Blue) << "Subdividing the blendshape meshes...";
      for(auto &Bi : B) {
        Bi.BuildHalfEdgeMesh();
        Bi.Subdivide();
      }
      stationary_indices = B00.filterVertices([=](const Vector3d& v) {
        return v[2] <= -0.45;
      });
      B00.BuildHalfEdgeMesh();
      B00.Subdivide();

      // Subdivide S, no need to update Sgrad because S will be deformed later
      ColorStream(ColorOutput::Blue) << "Subdividing the training meshes...";
      for(auto &Si : S) {
        Si.BuildHalfEdgeMesh();
        Si.Subdivide();
      }

      num_faces *= 4;
    }

    // The reconstructed mesh are updated using the new set of blendshape
    // weights, need to use laplacian deformation to refine them again
    #if 0
    ColorStream(ColorOutput::Blue) << "Updating the training shapes...";
    #pragma omp parallel for
    for(int i=0;i<num_poses;++i) {
      ColorStream(ColorOutput::Green) << "Updating training shape " << i << "...";
      auto& m = B00;
      MeshDeformer deformer;
      deformer.setSource(S[i]);

      vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
        Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
        return c[2] > -0.5;
      });
      deformer.setValidFaces(valid_faces);

      S[i] = deformer.deformWithPoints(point_clouds[i], PointCloud(), 5);
      S[i].Write( InBlendshapesDirectory("S_refined_" + to_string(i) + ".obj") );
    }
    #endif

    // compute deformation gradients for S
    for(int i=0;i<num_poses;++i) {
      Sgrad[i].resize(num_faces);
      for(int j=0;j<num_faces;++j) {
        // assign the reshaped gradient to the j-th row of Sgrad[i]
        Sgrad[i][j] = triangleGradient(S[i], j);
      }
    }
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B[i].Write( InBlendshapesDirectory("B_"+to_string(i)+".obj") );
  }

  // write out the blendshapes
  for(int i=0;i<num_poses;++i) {
    S[i].Write( InBlendshapesDirectory("S_"+to_string(i)+".obj") );
  }
}
