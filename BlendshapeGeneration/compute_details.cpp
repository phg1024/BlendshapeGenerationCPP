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

namespace {
  Eigen::MatrixXd load_matrix(const string& filename) {
    vector<string> lines = ReadFileByLine(filename);
    Eigen::MatrixXd mat;
    vector<vector<double>> elems;
    std::transform(lines.begin(), lines.end(), std::back_inserter(elems),
                   [](const string& line) {
                     vector<string> parts;
                     boost::algorithm::split(parts, line,
                                             boost::algorithm::is_any_of(" "),
                                             boost::algorithm::token_compress_on);
                     auto parts_end = std::remove_if(parts.begin(), parts.end(),
                                                     [](const string &s) {
                                                       return s.empty();
                                                     });
                     vector<double> row(std::distance(parts.begin(), parts_end));
                     std::transform(parts.begin(), parts_end, row.begin(),
                                    [](const string &s) {
                                      return std::stod(s);
                                    });
                     return row;
                   });
    const int nrows = elems.size(), ncols = elems.front().size();
    mat.resize(nrows, ncols);
    for(int i=0;i<nrows;++i) {
      for(int j=0;j<ncols;++j) {
        mat(i, j) = elems[i][j];
      }
    }
    return mat;
  }

  void save_matrix(ostream& os, const Eigen::MatrixXd& m, bool write_size=true) {
    if(write_size) os << m.rows() << ' ' << m.cols() << endl;
    for(int i=0;i<m.rows();++i) {
      for(int j=0;j<m.cols();++j) {
        os << m(i, j) << ' ';
      }
      os << endl;
    }
  }
}

class MeshDifferBase {
public:
  MeshDifferBase(const Eigen::VectorXd& wexp, const vector<double>& weights) : wexp(wexp), weights(weights) {}
  virtual void operator()(const BasicMesh& corase_mesh, const BasicMesh& detailed_mesh, const string& output_filename) = 0;
  virtual void write_diff(ostream& os) = 0;

  virtual void write(const string& filename) {
    cout << "Writing diff to " << filename << endl;
    ofstream fout(filename);
    write_wexp(fout);
    write_diff(fout);
  }

  void write_wexp(ostream& os) {
    cout << "Writing wexp ..." << endl;
    os << wexp.size() << endl;
    for(int i=0;i<wexp.size();++i) {
      os << wexp(i) << ' ';
    }
    os << endl;
  }
protected:
  Eigen::VectorXd wexp;
  vector<double> weights;
};

class VertexDiffer : public MeshDifferBase {
public:
  VertexDiffer(const Eigen::VectorXd& wexp, const vector<double>& weights) : MeshDifferBase(wexp, weights) {}
  void operator()(const BasicMesh& coarse_mesh, const BasicMesh& detailed_mesh, const string& output_filename) override {
    const int num_verts = coarse_mesh.NumVertices();
    assert(detailed_mesh.NumVertices() == num_verts);
    assert(weights.size() == num_verts);

    diffs = detailed_mesh.vertices() - coarse_mesh.vertices();

#if 0
    Eigen::Vector3d mean_diff = (diffs.colwise().sum() / num_verts).eval();

    cout << diffs.rows() << " vs " << weights.size() << endl;

    for(int i=0;i<diffs.rows();++i) {
      diffs.row(i) = (diffs.row(i) - mean_diff.transpose()); //* weights[i];
    }
#endif

    write(output_filename);
  }

  void write_diff(ostream& os) override {
    cout << "Writing diff ..." << endl;
    save_matrix(os, diffs);
  }

private:
  Eigen::MatrixXd diffs;
};

class NormalDiffer : public MeshDifferBase {
public:
  NormalDiffer(const Eigen::VectorXd& wexp, const vector<double>& weights) : MeshDifferBase(wexp, weights) {}
  void operator()(const BasicMesh& coarse_mesh, const BasicMesh& detailed_mesh, const string& output_filename) override {
    const int num_verts = coarse_mesh.NumVertices();
    assert(detailed_mesh.NumVertices() == num_verts);
    assert(weights.size() == num_verts);

    diffs.resize(num_verts, 3);
    for(int i=0;i<diffs.rows();++i) diffs.row(i) = detailed_mesh.vertex_normal(i) - coarse_mesh.vertex_normal(i);

    write(output_filename);
  }

  void write_diff(ostream& os) override {
    cout << "Writing normals ..." << endl;
    save_matrix(os, diffs);
  }
private:
  Eigen::MatrixXd diffs;
};

template <class MeshDiffer>
void compute_details_exp(
  const string& res_filename,
  const string& init_bs_path,
  const string& detailed_mesh_filename,
  const string& output_fitted_mesh_filename,
  const string& output_diff_filename
) {
  const string hair_mask_path("/home/phg/Data/Multilinear/hair_region_indices.txt");
  auto hair_region_indices_quad = LoadIndices(hair_mask_path);
  vector<int> hair_region_indices;
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  for(auto fidx : hair_region_indices_quad) {
    hair_region_indices.push_back(fidx*2);
    hair_region_indices.push_back(fidx*2+1);
  }
  // HACK: each valid face i becomes [4i, 4i+1, 4i+2, 4i+3] after the each
  // subdivision. See BasicMesh::Subdivide for details
  const int max_subdivisions = 1;
  for(int i=0;i<max_subdivisions;++i) {
    vector<int> hair_region_indices_new;
    for(auto fidx : hair_region_indices) {
      int fidx_base = fidx*4;
      hair_region_indices_new.push_back(fidx_base);
      hair_region_indices_new.push_back(fidx_base+1);
      hair_region_indices_new.push_back(fidx_base+2);
      hair_region_indices_new.push_back(fidx_base+3);
    }
    hair_region_indices = hair_region_indices_new;
  }
  unordered_set<int> hair_region_indices_set(hair_region_indices.begin(), hair_region_indices.end());

  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string mesh_filename(datapath + "Tester_1/Blendshape/shape_0.obj");

  const int num_blendshapes = 46;
  vector<BasicMesh> blendshapes(num_blendshapes+1);
#pragma omp parallel for
  for(int i=0;i<=num_blendshapes;++i) {
    blendshapes[i].LoadOBJMesh( init_bs_path + "/" + "B_" + to_string(i) + ".obj" );
    blendshapes[i].ComputeNormals();
  }

  BasicMesh m = blendshapes[0];

  auto recon_results = LoadReconstructionResult(res_filename);
  cout << "Recon results loaded." << endl;
  {
    MatrixX3d verts0 = blendshapes[0].vertices();
    MatrixX3d verts = verts0;
    for(int j=1;j<=num_blendshapes;++j) {
      verts += (blendshapes[j].vertices() - verts0) * recon_results.params_model.Wexp_FACS(j);
    }
    m.vertices() = verts;
    m.ComputeNormals();
  }

  m.Write(output_fitted_mesh_filename);

  BasicMesh detailed_mesh;
  detailed_mesh.LoadOBJMesh(detailed_mesh_filename);

  // Compute per-vertex weights
  unordered_set<int> hair_region_vertices;
  for(int i=0;i<m.NumFaces();++i) {
    if(hair_region_indices_set.count(i)) {
      auto face_i = m.face(i);
      hair_region_vertices.insert(face_i[0]);
      hair_region_vertices.insert(face_i[1]);
      hair_region_vertices.insert(face_i[2]);
    }
  }

  double max_y = -3, min_y = 3;
  for(auto vi : hair_region_vertices) {
    double y = m.vertex(vi)[1];
    max_y = max(max_y, y);
    min_y = min(min_y, y);
  }

  vector<double> vert_weights(m.NumVertices(), 1.0);
  for(auto vi : hair_region_vertices) {
    double y = m.vertex(vi)[1];
    vert_weights[vi] = (y - min_y) / (max_y - min_y);
  }

  // Compute the difference map
  MeshDiffer differ(recon_results.params_model.Wexp_FACS, vert_weights);
  differ(m, detailed_mesh, output_diff_filename);
}

template <class MeshDiffer>
void compute_details_exp(
  const string& res_filename,
  const string& detailed_mesh_filename,
  const string& coarse_mesh_file,
  const string& output_diff_filename
) {
  const string hair_mask_path("/home/phg/Data/Multilinear/hair_region_indices.txt");
  auto hair_region_indices_quad = LoadIndices(hair_mask_path);
  vector<int> hair_region_indices;
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  for(auto fidx : hair_region_indices_quad) {
    hair_region_indices.push_back(fidx*2);
    hair_region_indices.push_back(fidx*2+1);
  }
  // HACK: each valid face i becomes [4i, 4i+1, 4i+2, 4i+3] after the each
  // subdivision. See BasicMesh::Subdivide for details
  const int max_subdivisions = 1;
  for(int i=0;i<max_subdivisions;++i) {
    vector<int> hair_region_indices_new;
    for(auto fidx : hair_region_indices) {
      int fidx_base = fidx*4;
      hair_region_indices_new.push_back(fidx_base);
      hair_region_indices_new.push_back(fidx_base+1);
      hair_region_indices_new.push_back(fidx_base+2);
      hair_region_indices_new.push_back(fidx_base+3);
    }
    hair_region_indices = hair_region_indices_new;
  }
  unordered_set<int> hair_region_indices_set(hair_region_indices.begin(), hair_region_indices.end());

  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string mesh_filename(datapath + "Tester_1/Blendshape/shape_0.obj");

  BasicMesh m;
  m.LoadOBJMesh(coarse_mesh_file);
  m.ComputeNormals();

  auto recon_results = LoadReconstructionResult(res_filename);
  cout << "Recon results loaded." << endl;

  BasicMesh detailed_mesh;
  detailed_mesh.LoadOBJMesh(detailed_mesh_filename);
  detailed_mesh.ComputeNormals();

  // Compute per-vertex weights
  unordered_set<int> hair_region_vertices;
  for(int i=0;i<m.NumFaces();++i) {
    if(hair_region_indices_set.count(i)) {
      auto face_i = m.face(i);
      hair_region_vertices.insert(face_i[0]);
      hair_region_vertices.insert(face_i[1]);
      hair_region_vertices.insert(face_i[2]);
    }
  }

  double max_y = -3, min_y = 3;
  for(auto vi : hair_region_vertices) {
    double y = m.vertex(vi)[1];
    max_y = max(max_y, y);
    min_y = min(min_y, y);
  }

  vector<double> vert_weights(m.NumVertices(), 1.0);
  for(auto vi : hair_region_vertices) {
    double y = m.vertex(vi)[1];
    vert_weights[vi] = (y - min_y) / (max_y - min_y);
  }

  // Compute the difference map
  cout << "Computing diffs ..." << endl;
  MeshDiffer differ(recon_results.params_model.Wexp_FACS, vert_weights);
  differ(m, detailed_mesh, output_diff_filename);
}

void visualize_details(
  const string& res_filename,
  const string& coarse_mesh_filename,
  const string& pca_comp_filename,
  const string& mapping_matrix_filename,
  const string& output_detailed_mesh_file
) {
  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string mesh_filename(datapath + "Tester_1/Blendshape/shape_0.obj");

  BasicMesh m;
  m.LoadOBJMesh(coarse_mesh_filename);

  auto recon_results = LoadReconstructionResult(res_filename);
  cout << "Recon results loaded." << endl;

  // Load pca components
  Eigen::MatrixXd pca_components = load_matrix(pca_comp_filename);
  Eigen::MatrixXd mapping_matrix = load_matrix(mapping_matrix_filename);

  Eigen::VectorXd pca_coeffs = (mapping_matrix * recon_results.params_model.Wexp_FACS).eval();
  cout << pca_coeffs << endl;

  Eigen::VectorXd diff = (pca_components.transpose() * pca_coeffs).eval();
  m.vertices() += Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(diff.data(), m.NumVertices(), 3);
  m.Write(output_detailed_mesh_file);
}

void visualize_details_normal(const string& res_filename,
                              const string& coarse_mesh_filename,
                              const string& pca_comp_filename,
                              const string& mapping_matrix_filename,
                              const string& output_normals_file) {
  const string datapath("/home/phg/Data/FaceWarehouse_Data_0/");
  const string mesh_filename(datapath + "Tester_1/Blendshape/shape_0.obj");

  auto recon_results = LoadReconstructionResult(res_filename);
  cout << "Recon results loaded." << endl;

  BasicMesh m;
  m.LoadOBJMesh(coarse_mesh_filename);
  m.ComputeNormals();

  // Load pca components
  Eigen::MatrixXd pca_components = load_matrix(pca_comp_filename);
  Eigen::MatrixXd mapping_matrix = load_matrix(mapping_matrix_filename);

  Eigen::VectorXd pca_coeffs = (mapping_matrix * recon_results.params_model.Wexp_FACS).eval();
  cout << pca_coeffs << endl;

  Eigen::VectorXd diff = (pca_components.transpose() * pca_coeffs).eval();
  for(int i=0;i<m.NumVertices();++i) {
    auto nvi = m.vertex_normal(i);
    diff(i*3+0) += nvi[0];
    diff(i*3+1) += nvi[1];
    diff(i*3+2) += nvi[2];
  }

  {
    ofstream fout(output_normals_file);
    save_matrix(fout, diff, false);
  }
}

void printUsage() {
  cout << "TBA" << endl;
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

  int argidx = 1;
  string option = argv[argidx++];

  if( option == "-c" ) {
    string res_file = argv[argidx++];
    string detailed_mesh_file = argv[argidx++];
    string output_fitted_mesh_file = argv[argidx++];
    string output_difference_map = argv[argidx++];
    string init_bs_path = argv[argidx++];
    compute_details_exp<VertexDiffer>(res_file,
                                      init_bs_path,
                                      detailed_mesh_file,
                                      output_fitted_mesh_file,
                                      output_difference_map);
  } else if (option == "-v" ) {
    string res_file = argv[argidx++];
    string coarse_mesh_file = argv[argidx++];
    string output_detailed_mesh_file = argv[argidx++];
    string pca_comp_file = argv[argidx++];
    string mapping_matrix_file = argv[argidx++];
    visualize_details(res_file,
                      coarse_mesh_file,
                      pca_comp_file,
                      mapping_matrix_file,
                      output_detailed_mesh_file);
  } else if (option == "-cn") {
    string res_file = argv[argidx++];
    string detailed_mesh_file = argv[argidx++];
    string coarse_mesh_file = argv[argidx++];
    string output_normal_map = argv[argidx++];
    compute_details_exp<NormalDiffer>(res_file,
                                      detailed_mesh_file,
                                      coarse_mesh_file,
                                      output_normal_map);
  } else if (option == "-vn") {
    string res_file = argv[argidx++];
    string coarse_mesh_file = argv[argidx++];
    string output_normals_file = argv[argidx++];
    string pca_comp_file = argv[argidx++];
    string mapping_matrix_file = argv[argidx++];
    visualize_details_normal(res_file,
                             coarse_mesh_file,
                             pca_comp_file,
                             mapping_matrix_file,
                             output_normals_file);
  }

  return 0;
#endif
}
