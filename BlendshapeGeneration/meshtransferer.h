#ifndef MESHTRANSFERER_H
#define MESHTRANSFERER_H

#include "common.h"
#include "basicmesh.h"

class MeshTransferer
{
public:
  MeshTransferer();
  ~MeshTransferer();

  void setSource(const BasicMesh &src) { S = src; }
  void setTarget(const BasicMesh &tgt) { T = tgt; }

  BasicMesh transfer(const BasicMesh &A);

private:
  BasicMesh S, T;
};

#endif // MESHTRANSFERER_H
