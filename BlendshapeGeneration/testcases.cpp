#include "testcases.h"

#if 0
void TestCases::testMatrix()
{
  DenseMatrix A = DenseMatrix::random(5, 5);
  cout << "A = \n" << A << endl;
  auto B = A.inv();
  cout << "B = \n" << B << endl;
  cout << "Bt = \n" << B.transposed() << endl;
  cout << "A*B = \n" << A * B << endl;
  cout << "B*A = \n" << B * A << endl;
}

void TestCases::testSaprseMatrix()
{
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

void TestCases::testCeres() {

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
