#include "BasicMesh.h"

BasicMesh::BasicMesh()
{
}


BasicMesh::~BasicMesh()
{
}

void BasicMesh::load(const string &filename)
{
  PhGUtils::OBJLoader loader;
  loader.load(filename);

  // initialize the basic mesh
  auto V = loader.getVerts();

  int nverts = V.size();
  verts.resize(V.size(), 3);
  for (int i = 0,offset=0; i < nverts; ++i) {
    verts(offset) = V[i].x; ++offset;
    verts(offset) = V[i].y; ++offset;
    verts(offset) = V[i].z; ++offset;
  }

  int nfaces = 0;
  auto F = loader.getFaces();
  for (int i = 0; i < F.size(); ++i) {
    nfaces += F[i].v.size()-2;
  }

  faces.resize(nfaces, 3);
  // triangulate the mesh
  for (int i = 0, offset=0; i < F.size(); ++i) {
    for (int j = 1; j < F[i].v.size()-1; ++j) {
      faces(offset) = F[i].v[0]; ++offset;
      faces(offset) = F[i].v[j]; ++offset;
      faces(offset) = F[i].v[j+1]; ++offset;
    }
  }
}

void BasicMesh::write(const string &filename)
{
  string content;
  // write verts
  for (int i = 0,offset=0; i < verts.nrow; ++i) {
    content += "v ";
    content += to_string(verts(offset)) + " "; ++offset;
    content += to_string(verts(offset)) + " "; ++offset;
    content += to_string(verts(offset)) + "\n"; ++offset;
  }

  // write faces
  for (int i = 0, offset = 0; i < faces.nrow; ++i) {
    content += "f ";
    content += to_string(faces(offset) + 1) + " "; ++offset;
    content += to_string(faces(offset) + 1) + " "; ++offset;
    content += to_string(faces(offset) + 1) + "\n"; ++offset;
  }

  ofstream fout(filename);
  fout << content << endl;
  fout.close();
}
