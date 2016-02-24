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

void deformationTransfer(const string& s0_name = string(), const string& t0_name = string(),
                         const string& s_name = string(), const string& t_name = string()) {
  const string datapath = "/home/phg/Data/FaceWarehouse_Data_0/";

  BasicMesh S0;
  if(s0_name.empty())
    S0.LoadOBJMesh(datapath + "Tester_1/Blendshape/shape_0.obj");
  else
    S0.LoadOBJMesh(s0_name);

  BasicMesh T0;
  if(t0_name.empty())
    T0.LoadOBJMesh(datapath + "Tester_106/Blendshape/shape_0.obj");
  else
    T0.LoadOBJMesh(t0_name);

  // use deformation transfer to create an initial set of blendshapes
  MeshTransferer transferer;

  transferer.setSource(S0); // set source and compute deformation gradient
  transferer.setTarget(T0); // set target and compute deformation gradient

  // find the stationary set of verteces
  vector<int> stationary_indices = T0.filterVertices([=](const Vector3d& v) {
    return v[2] <= -0.45;
  });
  transferer.setStationaryVertices(stationary_indices);

  BasicMesh S;
  if(s_name.empty())
    S.LoadOBJMesh(datapath + "Tester_1/Blendshape/shape_22.obj");
  else
    S.LoadOBJMesh(s_name);

  BasicMesh T = transferer.transfer(S);
  if(t_name.empty())
    T.Write("transferred.obj");
  else
    T.Write(t_name);
}

void printUsage() {
  cout << "Deformation transfer: [program] -d" << endl;
}

int main(int argc, char *argv[])
{
  google::InitGoogleLogging(argv[0]);

#if RUN_TESTS
  TestCases::testCeres();
  return 0;
#else

  if( argc < 2 ) {
    printUsage();
  }

  if(argc > 4) {
    for(int i=1;i<4;++i)
      cout << argv[i] << endl;
    deformationTransfer(argv[1], argv[2], argv[3], argv[4]);
  } else {
    string option = argv[1];

    if( option == "-d") {
      deformationTransfer();
    }
  }

  return 0;
#endif
}