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
#include <MultilinearReconstruction/OffscreenMeshVisualizer.h>
#include <MultilinearReconstruction/meshvisualizer.h>

#include "meshdeformer.h"
#include "meshtransferer.h"
#include "cereswrapper.h"

#include "Geometry/matrix.hpp"
#include "triangle_gradient.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>

struct CombinedIdentityExpressionCostFunction {
  CombinedIdentityExpressionCostFunction(
    const MultilinearModel& model,
    const BasicMesh& target
  ) : model(model), target(target) {
  }

  bool operator()(const double *const *params, double* residual) const {
    VectorXd wid = Map<const VectorXd>(params[0], 50).eval();
    VectorXd wexp = Map<const VectorXd>(params[1], 25).eval();

    // Apply the weights to MM
    model.ApplyWeights(wid, wexp);

    auto tm = model.GetTM();
    const int num_residues = tm.size() / 3;

    // Compute the residues
    #pragma omp parallel for
    for(int i=0;i<num_residues;++i) {
      auto vi = target.vertex(i);

      //cout << vi << endl;
      //cout << tm(offset) << ", " << tm(++offset) << ", " << tm(++offset)

      double dx = vi[0] - tm(i*3);
      double dy = vi[1] - tm(i*3+1);
      double dz = vi[2] - tm(i*3+2);

      residual[i] = sqrt(dx*dx + dy*dy + dz*dz);
    }

    return true;
  }

  mutable MultilinearModel model;
  const BasicMesh& target;
};

void FitMesh(const string& meshfile, const string& output_mesh_filename) {
  // Load the target mesh
  BasicMesh mesh(meshfile);
  const string home_directory = QDir::homePath().toStdString();
  cout << "Home dir: " << home_directory << endl;

  // Load multilinear model
  MultilinearModel model(home_directory + "/Data/Multilinear/blendshape_core.tensor");
  MultilinearModelPrior model_prior;
  model_prior.load(home_directory + "/Data/Multilinear/blendshape_u_0_aug.tensor",
                   home_directory + "/Data/Multilinear/blendshape_u_1_aug.tensor");

  // Optimize for both identity and expression weights
  ceres::Problem problem;
  VectorXd wid = model_prior.Wid0;
  VectorXd wexp = model_prior.Wexp0;

  // Add per-vertex constraints
  ceres::DynamicNumericDiffCostFunction<CombinedIdentityExpressionCostFunction>
    *cost_function = new ceres::DynamicNumericDiffCostFunction<CombinedIdentityExpressionCostFunction>(
      new CombinedIdentityExpressionCostFunction(
        model,
        mesh
      )
    );
  cout << mesh.NumVertices() << endl;
  cost_function->AddParameterBlock(wid.size());
  cost_function->AddParameterBlock(wexp.size());
  cost_function->SetNumResiduals(mesh.NumVertices());
  problem.AddResidualBlock(cost_function, NULL, vector<double*>{wid.data(), wexp.data()});

  // Add prior terms

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = 30;
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    //options.minimizer_type = ceres::LINE_SEARCH;
    //options.line_search_direction_type = ceres::STEEPEST_DESCENT;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    summary.BriefReport();
  }

  // Output the final mesh
  model.ApplyWeights(wid, wexp);
  BasicMesh output_mesh = mesh;
  output_mesh.UpdateVertices(model.GetTM());
  output_mesh.Write(output_mesh_filename);

  // Output the fitted identity and expression
  {
    ofstream fout(output_mesh_filename + ".weights");
    auto write_vector = [](const VectorXd& v, ofstream& os) {
      os << v.rows() << ' ';
      for(int i=0;i<v.rows();++i) {
        os << v(i) << ' ';
      }
      os << "\n";
    };
    write_vector(wid, fout);
    write_vector(wexp, fout);
  }
}

void printUsage(const string& program_name) {
  cout << "Usage: " << program_name << " target_mesh output_mesh" << endl;
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);

  google::InitGoogleLogging(argv[0]);

  if( argc < 3 ) {
    printUsage(argv[0]);
  } else {
    FitMesh(argv[1], argv[2]);
    return 0;
  }
}
