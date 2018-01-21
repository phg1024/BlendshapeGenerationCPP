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
    blendshapes_subdivided = settings["blendshapes_subdivided"];
    mask_nose_and_fore_head = settings["mask_nose_and_fore_head"];
  } else {
    use_init_blendshapes = false;
    do_subdivision = false;
    blendshapes_subdivided = false;
    mask_nose_and_fore_head = false;
  }

  reporter = Reporter("Blendshape Refiner");

  {
    reporter.Tic("Initialization");

    model = MultilinearModel("/home/phg/Data/Multilinear/blendshape_core.tensor");

    model_prior.load("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                     "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");

    template_mesh = BasicMesh("/home/phg/Data/Multilinear/template.obj");

    {
      // Load indices
      auto load_indices = [](const string& indices_filename) {
        auto indices_quad = LoadIndices(indices_filename);
        // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
        unordered_set<int> indices;
        for(auto fidx : indices_quad) {
          indices.insert(fidx*2);
          indices.insert(fidx*2+1);
        }

        return indices;
      };

      hair_region_indices = load_indices("/home/phg/Data/Multilinear/hair_region_indices.txt");
      nose_forehead_indices = load_indices("/home/phg/Data/Multilinear/nose_and_forehead_indices.txt");
      extended_hair_region_indices = load_indices("/home/phg/Data/Multilinear/extended_hair_region_indices.txt");
    }

    reporter.Toc();
  }
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

void BlendshapeRefiner::SetExampleMeshesPath(const string &path){
  cout << "Setting example meshes path to " << path << endl;
  example_meshes_path = fs::path(path);
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
  reporter.Tic("Loading template meshes");

  A.resize(num_shapes + 1);

  for(int i=0;i<=num_shapes;++i) {
    A[i].LoadOBJMesh(path + basename + to_string(i) + ".obj");
  }

  reporter.Toc();
}

void BlendshapeRefiner::LoadInitialBlendshapes() {
  reporter.Tic("Loading initial blendshapes");
  Binit.resize(num_shapes + 1);

  for(int i=0;i<=num_shapes;++i) {
    Binit[i].LoadOBJMesh( (input_blendshapes_path / fs::path("B_" + to_string(i) + ".obj")).string() );
  }
  reporter.Toc();
}

void BlendshapeRefiner::LoadSelectionFile(const string& selection_filename) {
  selection_indices = LoadIndices((resources_path / fs::path(selection_filename)).string());
  cout << "selection indices loaded." << endl;
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
  reporter.Tic("Loading input reconstruction results");

  vector<pair<string, string>> image_points_filenames = ParseSettingsFile( (resources_path / fs::path(settings_filename)).string() );

  image_bundles.resize(selection_indices.size());
  int order = 0;
  for(auto& img_idx : selection_indices) {
    cout << img_idx << endl;

    fs::path image_filename = resources_path / fs::path(to_string(img_idx) + ".jpg");
    fs::path pts_filename = resources_path / fs::path(to_string(img_idx) + ".pts");
    fs::path res_filename = reconstructions_path / fs::path(to_string(img_idx) + ".jpg.res");
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string());
    auto recon_results = LoadReconstructionResult(res_filename.string());

    image_bundles[order++] = ImageBundle(to_string(img_idx) + ".jpg",
                                         image_points_pair.first,
                                         image_points_pair.second,
                                         recon_results);
  }
  //utils::pause();

  cout << selection_indices.size() << " reconstruction results loaded." << endl;

  // Set the number of poses
  num_poses = selection_indices.size();

  reporter.Toc();
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
  reporter.Tic("Loading input point clouds");

  // Load all points files
  point_clouds.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    int img_idx = selection_indices[i];
    int point_cloud_idx = img_idx;
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

    // Write this out so we can measure the reconstruction error later
    {
      ofstream fout( InBlendshapesDirectory("P_" + to_string(selection_indices[i]) + ".obj") );
      for(int j=0;j<point_clouds[i].rows();++j) {
        const auto& row_j = point_clouds[i].row(j);
        fout << "v " << row_j[0] << ' ' << row_j[1] << ' ' << row_j[2] << '\n';
      }
      fout.close();
    }

  }

  reporter.Toc();
}

void BlendshapeRefiner::LoadInputExampleMeshes() {
  reporter.Tic("Loading input example meshes");

  // Load all points files
  point_clouds.resize(num_poses);
  for(int i=0;i<num_poses;++i) {
    int img_idx = selection_indices[i];
    int point_cloud_idx = img_idx;
    assert(point_cloud_idx != -1);

    // Load the example mesh

    // Sample the mesh to get point clouds
    BasicMesh example_i( ( example_meshes_path / fs::path(to_string(point_cloud_idx)) / fs::path("deformed.obj") ).string() );

    // Load the normal constraints to get all the relevant faces
    set<int> pointcloud_faces;
    {
      string normals_constraints_filename =
        (example_meshes_path / fs::path(to_string(point_cloud_idx)) / fs::path("constraints.txt")).string();
      ifstream fin(normals_constraints_filename);
      while(fin) {
        int fidx;
        double bx, by, bz, nx, ny, nz;
        fin >> fidx >> bx >> by >> bz >> nx >> ny >> nz;
        pointcloud_faces.insert(fidx);
      }
    }

    point_clouds[i] = example_i.samplePoints(32, pointcloud_faces);

    // Write this out so we can measure the reconstruction error later
    {
      ofstream fout( InBlendshapesDirectory("P_" + to_string(selection_indices[i]) + ".obj") );
      for(int j=0;j<point_clouds[i].rows();++j) {
        const auto& row_j = point_clouds[i].row(j);
        fout << "v " << row_j[0] << ' ' << row_j[1] << ' ' << row_j[2] << '\n';
      }
      fout.close();
    }

  }

  reporter.Toc();
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
  reporter.Tic("Creating training shapes");

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
    S0[i].Write( InBlendshapesDirectory("Sinit_" + to_string(selection_indices[i]) + ".obj") );
  }
  ColorStream(ColorOutput::Blue)<< "initial training shapes created.";


  auto& m = (template_mesh);
  vector<int> valid_faces = m.filterFaces([&m](Vector3i fi) {
    Vector3d c = (m.vertex(fi[0]) + m.vertex(fi[1]) + m.vertex(fi[2]))/ 3.0;
    return c[2] > -0.5;
  });

  S.resize(num_poses);
  Slandmarks.resize(num_poses);
  #pragma omp parallel for
  for(int i=0;i<num_poses;++i) {
    ColorStream(ColorOutput::Green)<< "creating refined training shape " << i;
    MeshDeformer deformer;
    deformer.setSource(m);
    deformer.setValidFaces(valid_faces);
    deformer.setSource(S0[i]);

    // Initialize the anchor vertices
    auto& vindices_i = image_bundles[i].params.params_model.vindices;
    Slandmarks[i] = MatrixX3d(vindices_i.size(), 3);
    for(int j=0;j<vindices_i.size();++j) {
      Slandmarks[i].row(j) = S0[i].vertex(vindices_i[j]);
    }

    deformer.setLandmarks(vindices_i);
    S[i] = deformer.deformWithPoints(point_clouds[i], Slandmarks[i], 20);
    S[i].Write( InBlendshapesDirectory("S0_" + to_string(selection_indices[i]) + ".obj") );
  }
  ColorStream(ColorOutput::Blue)<< "refined training shapes created.";

  reporter.Toc();
}

void BlendshapeRefiner::InitializeBlendshapes() {
  reporter.Tic("Initializing blendshapes");

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
    Ti.Write( InBlendshapesDirectory("S0_verify_" + std::to_string(selection_indices[i]) + ".obj") );
  }

  reporter.Toc();
}

#define WEIGHT_ESTIMATION_USE_AUTODIFF_COST_FUNCTIONS 0

#if WEIGHT_ESTIMATION_USE_AUTODIFF_COST_FUNCTIONS
struct PointResidual {
  PointResidual(const Vector3d& v, int idx, const vector<MatrixX3d> &dB)
    : v(v), vidx(idx), dB(dB) {}

  template <typename T>
  bool operator()(const T* const alpha, T* residual) const {
    int nshapes = dB.size();
    T p[3] = {T(0), T(0), T(0)};
    // compute
    for(int i=0;i<nshapes;++i) {
      Vector3d vi = dB[i].row(vidx);
      p[0] += T(vi[0]) * alpha[i];
      p[1] += T(vi[1]) * alpha[i];
      p[2] += T(vi[2]) * alpha[i];
    }
    residual[0] = T(v[0]) - p[0];
    residual[1] = T(v[1]) - p[1];
    residual[2] = T(v[2]) - p[2];
    return true;
  }

private:
  const Vector3d v;
  const int vidx;
  const vector<MatrixX3d> &dB;
};
#else
struct PointResidual : public ceres::SizedCostFunction<3, 46> {
  PointResidual(const Vector3d& v, int idx, const vector<MatrixX3d> &dB)
    : v(v), vidx(idx), dB(dB) {}

  virtual bool Evaluate(double const * const *params,
                        double* residual,
                        double** jacobians) const {
    const double* alpha = params[0];
    const int nshapes = dB.size();
    Vector3d p(0, 0, 0);

    for(int i=0;i<nshapes;++i) {
      p += dB[i].row(vidx) * alpha[i];
    }
    Vector3d r = v - p;

    residual[0] = r[0];
    residual[1] = r[1];
    residual[2] = r[2];

    if( jacobians != NULL ) {
      if(jacobians[0] != NULL) {
        // r = v - \alpha * [v_0, v_1, ..., v_{n-1}]
        //   = v - \sum_{i=0}^{n-1} \alpha_i v_i

        // \frac{\partial r}{\partial \alpha_i} = -v_i

        for(int i=0;i<nshapes;++i) {
          // jacobians[0][i] = \frac{\partial r_0}{\partial \alpha_i}
          jacobians[0][i] = -dB[i].row(vidx)[0];

          // jacobians[0][i + nshapes] = \frac{\partial r_1}{\partial \alpha_i}
          jacobians[0][nshapes+i] = -dB[i].row(vidx)[1];

          // jacobians[0][i + nshapes*2] = \frac{\partial r_2}{\partial \alpha_i}
          jacobians[0][nshapes*2+i] = -dB[i].row(vidx)[2];
        }
      }
    }
    return true;
  }

private:
  const Vector3d v;
  const int vidx;
  const vector<MatrixX3d> &dB;
};
#endif

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

VectorXd BlendshapeRefiner::EstimateWeights(
  const BasicMesh &S,
  const BasicMesh &B0,
  const vector <MatrixX3d> &dB,
  const VectorXd &w0,
  const VectorXd &wp,
  double w_prior,
  int itmax,
  const vector<int>& valid_vertices
) {
  boost::timer::auto_cpu_timer t("Expression weights estimation finished in %w seconds.\n");
  reporter.Tic("Expression weights estimation");

  VectorXd w = w0;

  Problem problem;

  MatrixX3d dV = S.vertices() - B0.vertices();

  // add all constraints
  if (valid_vertices.empty()) {
    const int num_verts = S.NumVertices();
    for(int i=0;i<num_verts;++i) {
#if WEIGHT_ESTIMATION_USE_AUTODIFF_COST_FUNCTIONS
      CostFunction *costfun = new AutoDiffCostFunction<PointResidual, 3, 46>(
        new PointResidual(dV.row(i), i, dB)
      );
#else
      CostFunction *costfun = new PointResidual(dV.row(i), i, dB);
#endif
      problem.AddResidualBlock(costfun, NULL, w.data()+1);
    }
  } else {
    // add only the valid vertices for estimating weights
    for(auto i : valid_vertices) {
#if WEIGHT_ESTIMATION_USE_AUTODIFF_COST_FUNCTIONS
      CostFunction *costfun = new AutoDiffCostFunction<PointResidual, 3, 46>(
        new PointResidual(dV.row(i), i, dB)
      );
#else
      CostFunction *costfun = new PointResidual(dV.row(i), i, dB);
#endif
      problem.AddResidualBlock(costfun, NULL, w.data()+1);
    }
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

  reporter.Toc();

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
                                                       const vector<int> &stationary_vertices,
                                                       const vector<bool> &stationary_faces_set,
                                                       bool refine_neutral_only) {
  boost::timer::auto_cpu_timer t("Blendshapes refinement finished in %w seconds.\n");
  reporter.Tic("Blendshapes refinement");

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

  // Precompute A_data
  const double w_data = 1.0;
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

  Eigen::SparseMatrix<double> A_data(nrows_data, ncols_data);
  A_data.setFromTriplets(Adata_coeffs.begin(), Adata_coeffs.end());
  A_data.makeCompressed();
  Eigen::SparseMatrix<double> A_data_T = A_data.transpose();
  Eigen::SparseMatrix<double> A_dataTA_data = (A_data_T * A_data).pruned();

  ColorStream(ColorOutput::Blue) << "computing per-face deformation gradients.";
  vector<MatrixXd> M(nfaces, MatrixXd(nshapes+1, 9));

  {
    boost::timer::auto_cpu_timer t("Per-face deformation gradients optimization finished in %w seconds.\n");

    #pragma omp parallel for
      for(int j=0;j<nfaces;++j) {
        // skip if the face is stationary or it's in the hair region
        if (stationary_faces_set[j]) {
          M[j] = MatrixXd(nshapes + 1, 9);
          for(int k=0;k<9;++k) M[j](0, k) = B00grads[0].rowptr(j)[k];
          for(int i=1;i<=nshapes;++i) {
            for(int k=0;k<9;++k) M[j](i, k) = B00grads[i].rowptr(j)[k] - B00grads[0].rowptr(j)[k];
          }
          continue;
        }

        bool is_face_masked = (hair_region_indices.count(j)
                           || nose_forehead_indices.count(j)
                           || extended_hair_region_indices.count(j))
                           && mask_nose_and_fore_head;
        const double w_data_mask = (is_face_masked?0.0001:1.0);

        // b_data
        VectorXd b_data(nrows_data);
        for(int i=0;i<nposes;++i) {
          auto Sgrad_ij = Sgrad[i][j];
          for(int k=0;k<9;++k) b_data(i*9+k) = Sgrad_ij(k);
        }
        b_data *= w_data * w_data_mask;

        VectorXd A_dataTb_data = A_data_T * b_data;

        // A_reg
        vector<Tripletd> Areg_coeffs;
        const int estimated_Areg_usage = 2070;
        Areg_coeffs.reserve(estimated_Areg_usage);
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
        const int estimated_Asim_usage = 9;
        Asim_coeffs.reserve(estimated_Asim_usage);
        VectorXd b_sim;
        if(refine_neutral_only) {
          // Mask certain region to further constrain the deformation
          bool masked = (hair_region_indices.count(j)
                     || nose_forehead_indices.count(j)
                     || extended_hair_region_indices.count(j))
                     && mask_nose_and_fore_head;
          w_sim = 0.0001 * (masked?10000:1);
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

        // A = [A_data; A_reg; A_sim]
        // AtA = A_dataTA_data + A_regTA_reg + A_simTA_sim;

        Eigen::SparseMatrix<double> A(nrows_total, ncols_total);
        vector<Tripletd> A_coeffs(0);
        A_coeffs.reserve(Adata_coeffs.size() + Areg_coeffs.size() + Asim_coeffs.size());

        // A_dataTA_data is common, so move it outside the loop
        //A_coeffs.insert(A_coeffs.end(), Adata_coeffs.begin(), Adata_coeffs.end());
        A_coeffs.insert(A_coeffs.end(), Areg_coeffs.begin(), Areg_coeffs.end());
        A_coeffs.insert(A_coeffs.end(), Asim_coeffs.begin(), Asim_coeffs.end());

        //cout << Adata_coeffs.size() << ", " << Areg_coeffs.size() << ", " << Asim_coeffs.size() << endl;
        //cout << A_coeffs.size() << endl;

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

        // Add back A_dataTA_data, which is computed outside
        AtA += A_dataTA_data * w_data_mask;

        const double epsilon = 0.0;//1e-9;
        Eigen::SparseMatrix<double> eye(ncols_total, ncols_total);
        for(int j=0;j<ncols_total;++j) eye.insert(j, j) = epsilon;
        AtA += eye;

        CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(AtA);

        VectorXd b(nrows_total);
        // A_dataTb_data is computed earlier
        //b.topRows(nrows_data) = b_data;
        b.middleRows(nrows_data, nrows_reg) = b_reg;
        b.bottomRows(nrows_sim) = b_sim;

        // Add A_dataTb_data to Atb
        VectorXd Mv = solver.solve(A.transpose() * b + A_dataTb_data);
        if(solver.info()!=Success) {
          cerr << "Failed to solve A\\b." << endl;
          exit(-1);
        }

        // Store it into M, M is allocated outside the loop
        for(int i=0;i<=nshapes;++i) {
          M[j].row(i) = Mv.middleRows(i*9, 9).transpose();
        }
      }
      ColorStream(ColorOutput::Blue) << "done.";
  }

  // reconstruct the blendshapes now
  ColorStream(ColorOutput::Blue) << "reconstructing blendshapes.";
  MeshTransferer transferer;
  transferer.setSource(B00);
  transferer.setTarget(B00);
  transferer.setStationaryVertices(stationary_vertices);

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
  transferer.setStationaryVertices(stationary_vertices);

  // @FIXME try do deformation transfer on the new neutral shape instead of using the
  // computed triangle gradients.

  // recovery all other shapes
  for(int i=1;i<=nshapes;++i) {
    ColorStream(ColorOutput::Green)<< "reconstructing blendshapes " << i;
    if(refine_neutral_only) {
      // Apply the deformation gradients of A0->Ai on B0 to create Bi
      transferer.setSource(A[0]);
      B_new[i] = transferer.transfer(A[i]);
    } else {
      // Apply the optimized deformation gradients of B0->Bi on B0 to create Bi
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

  reporter.Toc();

  return B_new;
}

void BlendshapeRefiner::Refine(bool initialize_only, bool disable_neutral_opt) {
  // [Step 0]: load initial blendshapes if possible
  if(use_init_blendshapes) LoadInitialBlendshapes();

  // HACK subdivide template mesh if use_init_blendshapes
  if(blendshapes_subdivided) {
    template_mesh.BuildHalfEdgeMesh();
    template_mesh.Subdivide();
  }

  // [Step 1]: deform the inintial training shapes with the input point clouds
  CreateTrainingShapes();

  // [Step 2]: create a set of initial blendshapes using initial neutral face mesh and template blendshapes
  InitializeBlendshapes();

  // HACK for generating point clouds fittings
  if(initialize_only) return;

  // HACK subdivide A and S if use_init_blendshapes
  if(blendshapes_subdivided) {
    ColorStream(ColorOutput::Blue) << "Using init blendshapes. Sudivide template meshes and training shapes ...";
    ColorStream(ColorOutput::Blue) << "Subdividing the meshes...";
    // Subdivide every mesh
    // Subdivide A and update prior, w_prior, stationary_indices
    ColorStream(ColorOutput::Blue) << "Subdividing the template meshes...";
    for(auto &Ai : A) {
      Ai.BuildHalfEdgeMesh();
      Ai.Subdivide();
    }

    // No need to subdivide S since it's generated from subdivided template mesh
    #if 0
    // Subdivide S, no need to update Sgrad because S will be deformed later
    ColorStream(ColorOutput::Blue) << "Subdividing the training meshes...";
    for(auto &Si : S) {
      Si.BuildHalfEdgeMesh();
      Si.Subdivide();
    }
    #endif
  }

  // [Step 3]: blendshape refinement

  // [blendshape refinement] data preparation
  auto& A0 = A[0];
  int num_faces = A0.NumFaces();

  auto ComputerPrior = [=](const vector<BasicMesh>& A) {
    ColorStream(ColorOutput::Blue) << "Computing prior ...";
    auto& A0 = A[0];
    int num_faces = A0.NumFaces();
    int num_shapes = A.size()-1;
    const double kappa = 0.1;

    vector<PhGUtils::Matrix3x3d> MA0(num_faces);
    for(int j=0;j<num_faces;++j) {
      MA0[j] = triangleGradient(A0, j);
    }

    // Compute per-face prior weight using the mask
    // apply strong weights to these region
    vector<double> w_mask(num_faces, 1);
    const double BIG_WEIGHT = 10000.0;
    if(mask_nose_and_fore_head) {
      for(auto fidx : hair_region_indices) {
        //cout << fidx << endl;
        w_mask[fidx] = BIG_WEIGHT;
      }
      for(auto fidx : extended_hair_region_indices) {
        //cout << fidx << endl;
        w_mask[fidx] = BIG_WEIGHT;
      }
      for(auto fidx : nose_forehead_indices) {
        //cout << fidx << endl;
        w_mask[fidx] = BIG_WEIGHT;
      }
    } else {
      cout << "Not masking nose and fore head region." << endl;
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
        w_prior(i, j) = (1+MAij_norm)/pow(kappa+MAij_norm, 2.0) * w_mask[j];
      }
    }

    return make_pair(prior, w_prior);
  };

  // Computer prior from A
  reporter.Tic("Computing prior");
  vector<vector<PhGUtils::Matrix3x3d>> prior;
  MatrixXd w_prior;
  tie(prior, w_prior) = ComputerPrior(A);
  reporter.Toc();

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
  reporter.Tic("Misc preparation");
  vector<MatrixX3d> dB(num_shapes);
  for(int i=0;i<num_shapes;++i) {
    dB[i] = B[i+1].vertices() - B[0].vertices();
  }

  // make a copy of B0 for regularization
  BasicMesh B00 = B[0];
  vector<int> stationary_indices = B00.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  });

  vector<int> front_face_vertices = B00.filterVertices([=](const Vector3d& v) {
    return v[2] > -0.45;
  });

  vector<int> stationary_faces = B00.filterFaces([=](const Vector3i& f) {
    Vector3d center = (B00.vertex(f[0]) + B00.vertex(f[1]) + B00.vertex(f[2])) / 3.0;
    return center[2] <= -0.45;
  });

  // For faster test of stationary faces
  vector<bool> stationary_faces_set(B00.NumFaces(), false);
  for(auto i : stationary_faces) stationary_faces_set[i] = true;

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
  reporter.Toc();

  ColorStream(ColorOutput::Red)<< "initialization done.";

  // [blendshape refinement] Main loop
  bool converged = false;
  double ALPHA_THRES = 1e-6;
  double B_THRES = 1e-6;
  double beta_max = 0.05, beta_min = 0.01;
  double gamma_max = 0.01, gamma_min = 0.01;
  double eta_max = 1.0, eta_min = 0.1;
  int iters = 0;
  const int maxIters = 2;   // This will do (maxIters - 1) subdivisions
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
    vector<BasicMesh> B_new;
    if(disable_neutral_opt) B_new = B;
    else B_new = RefineBlendshapes(S, Sgrad, A, B, B00, alpha, beta, gamma, prior, w_prior, stationary_indices, stationary_faces_set);

    // Refine all blendshapes
    B_new = RefineBlendshapes(S, Sgrad, A, B_new, B_new[0], alpha, beta, gamma, prior, w_prior, stationary_indices, stationary_faces_set, false);

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
      alpha_new[i] = EstimateWeights(S[i], B[0], dB,
                                     alpha[i], alpha_init[i], eta, 2,
                                     front_face_vertices);
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

      S[i].Write( InBlendshapesDirectory("S_fitted_" + to_string(selection_indices[i]) + ".obj") );
    }

    if(iters == maxIters) break;

    // Optional: subdivide all meshes
    if(do_subdivision){
      reporter.Tic("Subdivision");

      // HACK: each valid face i becomes [4i, 4i+1, 4i+2, 4i+3] after the each
      // Update both hair_region_indices and nose_forehead_indices
      auto update_indices = [](const unordered_set<int>& in_indices) {
        vector<int> indices_new;
        for(auto fidx : in_indices) {
          int fidx_base = fidx*4;
          indices_new.push_back(fidx_base);
          indices_new.push_back(fidx_base+1);
          indices_new.push_back(fidx_base+2);
          indices_new.push_back(fidx_base+2);
        }
        return unordered_set<int>(indices_new.begin(), indices_new.end());
      };
      hair_region_indices = update_indices(hair_region_indices);
      nose_forehead_indices = update_indices(nose_forehead_indices);
      extended_hair_region_indices = update_indices(extended_hair_region_indices);

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

      reporter.Toc();
    }

    // The reconstructed mesh are updated using the new set of blendshape
    // weights, need to use laplacian deformation to refine them again
    ColorStream(ColorOutput::Blue) << "Updating the training shapes...";
    reporter.Tic("Update training shapes");
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
      deformer.setLandmarks(image_bundles[i].params.params_model.vindices);

      S[i] = deformer.deformWithPoints(point_clouds[i], Slandmarks[i], 5);
      S[i].Write( InBlendshapesDirectory("S_refined_" + to_string(selection_indices[i]) + ".obj") );
    }

    // compute deformation gradients for S
    for(int i=0;i<num_poses;++i) {
      Sgrad[i].resize(num_faces);
      for(int j=0;j<num_faces;++j) {
        // assign the reshaped gradient to the j-th row of Sgrad[i]
        Sgrad[i][j] = triangleGradient(S[i], j);
      }
    }
    reporter.Toc();
  }

  // write out the blendshapes
  for(int i=0;i<num_shapes+1;++i) {
    B[i].Write( InBlendshapesDirectory("B_"+to_string(i)+".obj") );
  }

  // write out the final refined example poses
  for(int i=0;i<num_poses;++i) {
    S[i].Write( InBlendshapesDirectory("S_"+to_string(selection_indices[i])+".obj") );
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

      S[i].Write( InBlendshapesDirectory("S_fitted_" + to_string(selection_indices[i]) + ".obj") );
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
    S[i].Write( InBlendshapesDirectory("S_"+to_string(selection_indices[i])+".obj") );
  }
}
