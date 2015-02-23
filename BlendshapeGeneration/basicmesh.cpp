#include "basicmesh.h"

BasicMesh::BasicMesh()
{
}


BasicMesh::~BasicMesh()
{
}

PointCloud BasicMesh::samplePoints(int points_per_face, double zcutoff) const
{
  int npoints = 0;
  vector<int> validfaces;

  for (int i = 0; i < faces.nrow; ++i) {
    // sample 8 points per face
    int fidx = i * 3;
    int v1 = faces(fidx), v2 = faces(fidx+1), v3 = faces(fidx+2);
    double z1 = verts(v1*3+2), z2 = verts(v2*3+2), z3 = verts(v3*3+2);
    double zc = (z1 + z2 + z3) / 3.0;
    if (zc > zcutoff) {
      npoints += points_per_face;
      validfaces.push_back(i);
    }
  }
  cout << "npoints = " << npoints << endl;
  PointCloud P;
  P.points.resize(npoints, 3);
  for (int i = 0, offset=0; i < validfaces.size(); ++i) {
    int fidx = validfaces[i] * 3;
    int v1 = faces(fidx), v2 = faces(fidx+1), v3 = faces(fidx+2);
    double x1 = verts(v1*3), x2 = verts(v2*3), x3 = verts(v3*3);
    double y1 = verts(v1*3+1), y2 = verts(v2*3+1), y3 = verts(v3*3+1);
    double z1 = verts(v1*3+2), z2 = verts(v2*3+2), z3 = verts(v3*3+2);

    for(int j=0;j<points_per_face;++j) {
        // sample a point
        double alpha = rand()/(double)RAND_MAX,
            beta = rand()/(double)RAND_MAX * (1-alpha),
            gamma = 1.0 - alpha - beta;

        auto Pptr = P.points.rowptr(offset); ++offset;
        Pptr[0] = x1*alpha + x2*beta + x3*gamma;
        Pptr[1] = y1*alpha + y2*beta + y3*gamma;
        Pptr[2] = z1*alpha + z2*beta + z3*gamma;
      }
    }
  cout << "points sampled." << endl;
  return P;
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
