#include "blendshapegeneration.h"
#include <QFileDialog>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

BlendshapeVisualizer::BlendshapeVisualizer(QWidget *parent):
  GL3DCanvas(parent)
{
  setSceneScale(1.5);
  setProjectionMode(GL3DCanvas::ORTHONGONAL);
  mouseInteractionMode = GL3DCanvas::VIEW_TRANSFORM;
}

BlendshapeVisualizer::~BlendshapeVisualizer()
{

}

void BlendshapeVisualizer::loadMesh(const string &filename)
{
  mesh.LoadOBJMesh(filename);
  mesh.ComputeNormals();
  computeDistance();
  repaint();
}

void BlendshapeVisualizer::loadReferenceMesh(const string &filename)
{
  refmesh.LoadOBJMesh(filename);
  refmesh.ComputeNormals();
  computeDistance();
  repaint();
}

void BlendshapeVisualizer::paintGL()
{
  GL3DCanvas::paintGL();

  glEnable(GL_DEPTH_TEST);

  enableLighting();
  if( mesh.NumFaces() > 0 ) {
    if( refmesh.NumFaces() > 0 ) drawMeshWithColor(mesh);
    else drawMesh(mesh);
  }
  disableLighting();

  if( !dists.empty() ) {
    #if 0
    double minVal = (*std::min_element(dists.begin(), dists.end()));
    double maxVal = (*std::max_element(dists.begin(), dists.end()));
    #else
    double minVal = 0.0;
    double maxVal = 0.150;
    #endif
    string minStr = "min: " + to_string(minVal);
    string maxStr = "max: " + to_string(maxVal);
    glColor4f(0, 0, 1, 1);
    renderText(25, 25, QString(minStr.c_str()));
    glColor4f(1, 0, 0, 1);
    renderText(25, 45, QString(maxStr.c_str()));

    drawColorBar(minVal, maxVal);
  }
}

void BlendshapeVisualizer::drawColorBar(double minVal, double maxVal) {
  // setup 2D view
  glViewport(0, 0, width(), height());
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width(), height(), 0, -1, 1);

  int nsegments = 128;

  double left = 0.85 * width();
  double w = 0.025 * width();
  double top = 0.25 * height();
  double h = 0.5 * height() / (nsegments-1);

  vector<QColor> colors(nsegments);
  for(int i=0;i<nsegments;++i) {
    double c = i / float(nsegments-1);
    double hval0 = c;
    colors[i] = QColor::fromHsvF(hval0*0.67, 1.0, 1.0);
  }

  for(int i=0;i<nsegments-1;++i) {
    glBegin(GL_QUADS);
    glNormal3f(0, 0, -1);

    glColor4f(colors[i].redF(), colors[i].greenF(), colors[i].blueF(), 1.0);
    float hstart = i * h;
    float hend = hstart + h;
    glVertex3f(left, top+hstart, -0.5); glVertex3f(left + w, top+hstart, -0.5);

    glColor4f(colors[i+1].redF(), colors[i+1].greenF(), colors[i+1].blueF(), 1.0);
    glVertex3f(left + w, top + hend, -0.5); glVertex3f(left, top + hend, -0.5);

    glEnd();
  }

  // draw texts
  glColor4f(0, 0, 0, 1);
  int ntexts = 6;
  for(int i=0;i<ntexts;++i) {
    double ratio = 1.0 - i / (float)(ntexts-1);
    double hpos = i * (0.5 * height()) / (ntexts - 1);
    string str = to_string(minVal + (maxVal - minVal) * ratio);
    renderText(left + w, top+hpos + 5.0, -0.5, QString(str.c_str()));
  }
}

void BlendshapeVisualizer::enableLighting()
{
  GLfloat light_position[] = {10.0, 4.0, 10.0,1.0};
  GLfloat mat_specular[] = {0.5, 0.5, 0.5, 1.0};
  GLfloat mat_diffuse[] = {0.375, 0.375, 0.375, 1.0};
  GLfloat mat_shininess[] = {25.0};
  GLfloat light_ambient[] = {0.05, 0.05, 0.05, 1.0};
  GLfloat white_light[] = {1.0, 1.0, 1.0, 1.0};

  //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
  //glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

  light_position[0] = -10.0;
  glLightfv(GL_LIGHT1, GL_POSITION, light_position);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
  //glLightfv(GL_LIGHT1, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
}

void BlendshapeVisualizer::disableLighting()
{
  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHTING);
}

// TODO Add a method to compute and visualize point set to surface distance
void BlendshapeVisualizer::computeDistance()
{
  if( refmesh.NumFaces() <= 0 ) return;

  int nfaces = refmesh.NumFaces();

  std::vector<Triangle> triangles;
  triangles.reserve(nfaces);
  for(int i=0,ioffset=0;i<nfaces;++i) {
    auto face_i = refmesh.face(i);
    int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
    auto p1 = refmesh.vertex(v1), p2 = refmesh.vertex(v2), p3 = refmesh.vertex(v3);
    Point a(p1[0], p1[1], p1[2]);
    Point b(p2[0], p2[1], p2[2]);
    Point c(p3[0], p3[1], p3[2]);

    triangles.push_back(Triangle(a, b, c));
  }

  Tree tree(triangles.begin(), triangles.end());
  tree.accelerate_distance_queries();

  dists.resize(mesh.NumVertices());
  for(int i=0;i<mesh.NumVertices();++i) {
    auto point_i = mesh.vertex(i);
    double px = point_i[0],
           py = point_i[1],
           pz = point_i[2];
    Tree::Point_and_primitive_id bestHit = tree.closest_point_and_primitive(Point(px, py, pz));
    double qx = bestHit.first.x(),
           qy = bestHit.first.y(),
           qz = bestHit.first.z();

    auto ref_face = *bestHit.second;
    auto ref_normal = CGAL::normal(ref_face[0], ref_face[1],  ref_face[2]);
    Vector3d normal0(ref_normal.x(), ref_normal.y(), ref_normal.z());

    auto normal = mesh.vertex_normal(i);

    if(normal.dot(normal0) < 0) {
      dists[i] = -1;
    } else {
      double dx = px - qx, dy = py - qy, dz = pz - qz;
      dists[i] = sqrt(dx*dx+dy*dy+dz*dz);
      if(dists[i] > 0.1) dists[i] = -1;
    }
  }
}

void BlendshapeVisualizer::drawMesh(const BasicMesh &m)
{
  glColor4f(0.75, 0.75, 0.75, 1.0);
  glBegin(GL_TRIANGLES);
  for(int i=0;i<m.NumFaces();++i) {
    auto face_i = m.face(i);
    int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
    auto norm_i = m.normal(i);
    glNormal3d(norm_i[0], norm_i[1], norm_i[2]);
    auto p1 = m.vertex(v1), p2 = m.vertex(v2), p3 = m.vertex(v3);

    glVertex3d(p1[0], p1[1], p1[2]);
    glVertex3d(p2[0], p2[1], p2[2]);
    glVertex3d(p3[0], p3[1], p3[2]);
  }
  glEnd();
}

void BlendshapeVisualizer::drawMeshWithColor(const BasicMesh &m)
{
  #if 0
  double maxVal = *(std::max_element(dists.begin(), dists.end()));
  double minVal = *(std::min_element(dists.begin(), dists.end()));
  #else
  double maxVal = 0.150;
  double minVal = 0.0;
  #endif

  glColor4f(0.75, 0.75, 0.75, 1.0);
  glBegin(GL_TRIANGLES);
  for(int i=0;i<m.NumFaces();++i) {
    auto face_i = m.face(i);
    int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];

    //auto norm_i = m.normal(i);
    //glNormal3d(norm_i[0], norm_i[1], norm_i[2]);

    auto p1 = m.vertex(v1), p2 = m.vertex(v2), p3 = m.vertex(v3);

    auto draw_vertex = [&](double dval, decltype(p1) pj, int vj) {
      double hval0 = 1.0 - max(min((dval-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
      if(dval < 0) {
        hval0 = 0.0;
      }
      QColor c0 = QColor::fromHsvF(hval0*0.67, 1.0, 1.0);
      float colors0[4] = {(float)c0.redF(), (float)c0.greenF(), (float)c0.blueF(), 1.0f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors0);
      auto nj = m.vertex_normal(vj);
      glNormal3d(nj[0], nj[1], nj[2]);
      glVertex3d(pj[0], pj[1], pj[2]);
    };

#if 1

    draw_vertex(dists[face_i[0]], p1, v1);
    draw_vertex(dists[face_i[1]], p2, v2);
    draw_vertex(dists[face_i[2]], p3, v3);

#else
    double hval0 = 1.0 - max(min((dists[face_i[0]]-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
    if(dists[face_i[0]] < 0) {
      hval0 = 1.0;
    }
    QColor c0 = QColor::fromHsvF(hval0*0.67, 1.0, 1.0);
    float colors0[4] = {(float)c0.redF(), (float)c0.greenF(), (float)c0.blueF(), 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors0);
    auto n1 = m.vertex_normal(v1);
    glNormal3d(n1[0], n1[1], n1[2]);
    glVertex3d(p1[0], p1[1], p1[2]);

    double hval1 = 1.0 - max(min((dists[face_i[1]]-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
    if(dists[face_i[0]] < 0) {
      hval1 = 1.0;
    }
    QColor c1 = QColor::fromHsvF(hval1*0.67, 1.0, 1.0);
    float colors1[4] = {(float)c1.redF(), (float)c1.greenF(), (float)c1.blueF(), 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors1);
    auto n2 = m.vertex_normal(v2);
    glNormal3d(n2[0], n2[1], n2[2]);
    glVertex3d(p2[0], p2[1], p2[2]);

    double hval2 = 1.0 - max(min((dists[face_i[2]]-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
    if(dists[face_i[0]] < 0) {
      hval2 = 1.0;
    }
    QColor c2 = QColor::fromHsvF(hval2*0.67, 1.0, 1.0);
    float colors2[4] = {(float)c2.redF(), (float)c2.greenF(), (float)c2.blueF(), 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors2);
    auto n3 = m.vertex_normal(v3);
    glNormal3d(n3[0], n3[1], n3[2]);
    glVertex3d(p3[0], p3[1], p3[2]);
#endif
  }
  glEnd();
}

BlendshapeGeneration::BlendshapeGeneration(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    canvas = new BlendshapeVisualizer(this);
    setCentralWidget(canvas);

    connect(ui.actionLoad_Mesh, SIGNAL(triggered()), this, SLOT(slot_loadMesh()));
    connect(ui.actionLoad_Reference, SIGNAL(triggered()), this, SLOT(slot_loadReferenceMesh()));
}

BlendshapeGeneration::~BlendshapeGeneration()
{

}

void BlendshapeGeneration::slot_loadMesh()
{
  QString filename = QFileDialog::getOpenFileName();
  canvas->loadMesh(filename.toStdString());
}

void BlendshapeGeneration::slot_loadReferenceMesh()
{
  QString filename = QFileDialog::getOpenFileName();
  canvas->loadReferenceMesh(filename.toStdString());
}
