#ifndef MESHTRANSFERER_H
#define MESHTRANSFERER_H

#include "common.h"
#include "basicmesh.h"

class MeshTransferer
{
public:
  MeshTransferer();
  ~MeshTransferer();

  void setSource(const BasicMesh &src) { S0 = src; }
  void setTarget(const BasicMesh &tgt) { T0 = tgt; }
  void setStationaryVertices(const vector<int> &sv) { stationary_vertices = sv; }

  BasicMesh transfer(const BasicMesh &A);

private:
  BasicMesh S0, T0;
  vector<int> stationary_vertices;
};

#endif // MESHTRANSFERER_H
