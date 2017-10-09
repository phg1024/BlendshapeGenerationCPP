#include "blendshapegeneration.h"
#include <QFileDialog>
#include <QImage>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLPaintDevice>

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

OffscreenBlendshapeVisualizer::OffscreenBlendshapeVisualizer(int w, int h)
  : use_side_view(false), canvasW(w), canvasH(h), has_texture(false) {
  // Load and Parse rendering settings
  {
    std::ifstream fin("/home/phg/Data/Settings/blendshape_vis.json");
    fin >> rendering_settings;
    //cout << rendering_settings << endl;
  }

  SetBasicRenderingParams();
}
OffscreenBlendshapeVisualizer::~OffscreenBlendshapeVisualizer() {}

void OffscreenBlendshapeVisualizer::SetBasicRenderingParams() {
  // Basic setup
  sceneScale = rendering_settings["scene_scale"];
  cameraPos = QVector3D(
    rendering_settings["campos"][0],
    rendering_settings["campos"][1],
    rendering_settings["campos"][2]);

  if(string(rendering_settings["mode"]) == "perspective") {
    projectionMode = PERSPECTIVE;
  } else {
    projectionMode = ORTHONGONAL;
  }
}

void OffscreenBlendshapeVisualizer::loadMesh(const string& filename) {
  mesh.LoadOBJMesh(filename);
  mesh.ComputeNormals();
  computeDistance();
}

void OffscreenBlendshapeVisualizer::loadReferenceMesh(const string& filename) {
  refmesh.LoadOBJMesh(filename);
  refmesh.ComputeNormals();
  computeDistance();
}

namespace {
BasicMesh AlignMesh(const BasicMesh& source, const BasicMesh& target) {
  BasicMesh aligned = source;

  bool use_face_region = true;
  vector<int> valid_faces;
  if(!use_face_region) {
    valid_faces.resize(target.NumFaces());
    for(int i=0;i<target.NumFaces();++i) valid_faces[i] = i;
  } else {
    auto valid_faces_quad = LoadIndices("/home/phg/Data/Multilinear/face_region_indices.txt");
    for(auto fidx : valid_faces_quad) {
      valid_faces.push_back(fidx*2);
      valid_faces.push_back(fidx*2+1);
    }
  }

  // Do ICP
  // Create the query tree for the valid faces
  const int nfaces = valid_faces.size();
  std::vector<Triangle> triangles;
  triangles.reserve(nfaces);

  {
    boost::timer::auto_cpu_timer t("[mesh alignment] data preparation time = %w seconds.\n");
    for(int i=0,ioffset=0;i<nfaces;++i) {
      auto face_i = target.face(valid_faces[i]);
      int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
      auto p1 = target.vertex(v1), p2 = target.vertex(v2), p3 = target.vertex(v3);
      Point a(p1[0], p1[1], p1[2]);
      Point b(p2[0], p2[1], p2[2]);
      Point c(p3[0], p3[1], p3[2]);

      triangles.push_back(Triangle(a, b, c));
    }
  }

  Tree tree(triangles.begin(), triangles.end());
  tree.accelerate_distance_queries();

  const int max_iters = 5;
  // Do a few iterations to optimize for R and t

  int iters = 0;
  while(iters < max_iters) {
    ++iters;
    aligned.ComputeNormals();

    vector<pair<Vector3d, Vector3d>> correspondence;
    for(int i=0;i<aligned.NumVertices();++i) {
      auto point_i = aligned.vertex(i);
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

      auto normal = aligned.vertex_normal(i);

      double weight_i = 1.0;
      if(normal.dot(normal0) < 0.) {
        continue;
      } else {
        double dx = px - qx, dy = py - qy, dz = pz - qz;
        double dist_i = sqrt(dx*dx+dy*dy+dz*dz);
        if(dist_i > 0.1) {
          continue;
        } else {
          correspondence.push_back(make_pair(point_i, Vector3d(qx, qy, qz)));
        }
      }
    }

    // Solve for R and t using the correspondence
    const int nconstraints = correspondence.size();
    cout << iters << ": " << nconstraints << endl;
    MatrixX3d P, Q;
    P.resize(nconstraints, 3); Q.resize(nconstraints, 3);
    for(int i=0;i<nconstraints;++i) {
      Vector3d p_i, q_i;
      tie(p_i, q_i) = correspondence[i];
      P.row(i) = p_i; Q.row(i) = q_i;
    }
    Vector3d pbar = P.colwise().sum() / nconstraints,
             qbar = Q.colwise().sum() / nconstraints;
    MatrixX3d X = P - pbar.replicate(1, nconstraints).transpose(),
             Y = Q - qbar.replicate(1, nconstraints).transpose();

    MatrixXd S = X.transpose() * Y;
    cout << S << endl;
    JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);
    auto sigma = svd.singularValues();
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    MatrixXd VdUt = V * U.transpose();
    double detVU = VdUt.determinant();
    Matrix3d eye3 = Matrix3d::Identity();
    eye3(2, 2) = detVU;
    Matrix3d R = V * eye3 * U.transpose();
    Vector3d t = qbar - R * pbar;

    // Apply R and t to the vertices of aligned
    aligned.vertices() =
      ((R * aligned.vertices().transpose() + t.replicate(1, aligned.NumVertices())).transpose()).eval();
  }

  return aligned;
}
}

void OffscreenBlendshapeVisualizer::computeDistance() {
  boost::timer::auto_cpu_timer t("Distance computation time = %w seconds.\n");
  if( refmesh.NumFaces() <= 0 ) return;

  int nfaces = refmesh.NumFaces();

  std::vector<Triangle> triangles;
  triangles.reserve(nfaces);

  {
    boost::timer::auto_cpu_timer t("[compute distance] data preparation time = %w seconds.\n");
    for(int i=0,ioffset=0;i<nfaces;++i) {
      auto face_i = refmesh.face(i);
      int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
      auto p1 = refmesh.vertex(v1), p2 = refmesh.vertex(v2), p3 = refmesh.vertex(v3);
      Point a(p1[0], p1[1], p1[2]);
      Point b(p2[0], p2[1], p2[2]);
      Point c(p3[0], p3[1], p3[2]);

      triangles.push_back(Triangle(a, b, c));
    }
  }

  Tree tree(triangles.begin(), triangles.end());
  tree.accelerate_distance_queries();

  if(align_mesh) mesh = AlignMesh(mesh, refmesh);
  // HACK if we align mesh, we are computing error for the face region only
  bool use_face_region = align_mesh;
  vector<int> valid_faces;
  if(!use_face_region) {
    valid_faces.resize(refmesh.NumFaces());
    for(int i=0;i<refmesh.NumFaces();++i) valid_faces[i] = i;
  } else {
    auto valid_faces_quad = LoadIndices("/home/phg/Data/Multilinear/face_region_indices.txt");
    for(auto fidx : valid_faces_quad) {
      valid_faces.push_back(fidx*2);
      valid_faces.push_back(fidx*2+1);
    }
  }
  set<int> valid_vertices;
  if(use_face_region) {
    for(auto fidx : valid_faces) {
      auto f_i = mesh.face(fidx);
      valid_vertices.insert(f_i[0]);
      valid_vertices.insert(f_i[1]);
      valid_vertices.insert(f_i[2]);
    }
  } else {
    for(int i=0;i<mesh.NumVertices();++i) valid_vertices.insert(i);
  }

  dists.resize(mesh.NumVertices(), 0);
  {
    boost::timer::auto_cpu_timer t("[compute distance] closest point search time = %w seconds.\n");
    #pragma omp parallel for
    for(int i=0;i<mesh.NumVertices();++i) {
      if(!valid_vertices.count(i)) {
        continue;
      }
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
        dists[i] = -dists[i];
      } else {
        double dx = px - qx, dy = py - qy, dz = pz - qz;
        dists[i] = sqrt(dx*dx+dy*dy+dz*dz);
        if(dists[i] > 0.1) dists[i] = -dists[i];
      }
    }
  }
}

void OffscreenBlendshapeVisualizer::setupViewing() {
  glViewport(0, 0, canvasW, canvasH);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

	switch(projectionMode)
	{
	case ORTHONGONAL:
		{
			glOrtho (-1.0 * sceneScale, 1.0 * sceneScale,
				-1.0 * sceneScale, 1.0 * sceneScale,
				-1.0 * sceneScale, 1.0 * sceneScale);
			break;
		}
	case FRUSTUM:
		{
			glFrustum(-1.0, 1.0, -1.0, 1.0, 1.0, 2.0);
			break;
		}
	case PERSPECTIVE:
		{
			gluPerspective(25.0, (float)width()/(float)height(), 0.01f, sceneScale * 10.0);
			break;
		}
	default:
		break;
	}

	glMatrixMode (GL_MODELVIEW);

	glLoadIdentity ();

	if(projectionMode==PERSPECTIVE)
	{
		//QVector3D scaledCameraPos = cameraPos / trackBall.getScale();
    QVector3D scaledCameraPos = cameraPos;
		gluLookAt(
			scaledCameraPos.x(), scaledCameraPos.y(), scaledCameraPos.z(),
			0.0, 0.0, 0.0,
			0.0, 1.0, 0.0);
	}
}

void OffscreenBlendshapeVisualizer::paint()
{
  boost::timer::auto_cpu_timer t("render time = %w seconds.\n");

  // Create a offscreen surface for drawing
  QSurfaceFormat format;
  format.setMajorVersion(3);
  format.setMinorVersion(3);

  QOffscreenSurface surface;
  surface.setFormat(format);
  surface.create();

  QOpenGLContext context;
  context.setFormat(format);
  if (!context.create()) {
    qFatal("Cannot create the requested OpenGL context!");
  }
  context.makeCurrent(&surface);

  const QRect drawRect(0, 0, canvasW, canvasH);
  const QSize drawRectSize = drawRect.size();

  QOpenGLFramebufferObjectFormat fboFormat;
  fboFormat.setSamples(16);
  fboFormat.setAttachment(QOpenGLFramebufferObject::Depth);

  QOpenGLFramebufferObject fbo(drawRectSize, fboFormat);
  fbo.bind();

  setupViewing();

  glEnable(GL_DEPTH_TEST);

  enableLighting();
  if(use_side_view) {
    glPushMatrix();
    glRotatef(-90, 0, 1, 0);
  }

  if( mesh.NumFaces() > 0 ) {
    if( refmesh.NumFaces() > 0 ) drawMeshWithColor(mesh);
    else drawMesh(mesh);
  } else {
    cout << "Drawing mesh ..." << endl;
    glDepthFunc(GL_LEQUAL);
    drawMesh(refmesh);

    cout << "Drawing vertices mesh ..." << endl;
    glDepthFunc(GL_ALWAYS);
    drawMeshVerticesWithColor(mesh);
  }

  if(use_side_view) {
    glPopMatrix();
  }

  disableLighting();

  if( !dists.empty() ) {
    #if 0
    double minVal = (*std::min_element(dists.begin(), dists.end()));
    double maxVal = (*std::max_element(dists.begin(), dists.end()));
    #else
    double minVal = 0.0;
    double maxVal = error_range;
    #endif
    string minStr = "min: " + to_string(minVal);
    string maxStr = "max: " + to_string(maxVal);

    drawColorBar(minVal, maxVal);

    QPainter painter;
    QOpenGLPaintDevice device(drawRectSize);
    painter.begin(&device);
    painter.setRenderHints(QPainter::Antialiasing | QPainter::HighQualityAntialiasing);

    painter.setPen(Qt::blue);
    painter.setBrush(Qt::blue);
    painter.drawText(25, 25, QString(minStr.c_str()));

    painter.setPen(Qt::red);
    painter.setBrush(Qt::red);
    painter.drawText(25, 45, QString(maxStr.c_str()));


    // draw texts for the color bar
    painter.setPen(Qt::black);
    painter.setBrush(Qt::black);
    {
      double left = 0.85 * width();
      double w = 0.025 * width();
      double top = 0.25 * height();
      //double h = 0.5 * height() / (nsegments-1);
      int ntexts = 6;
      for(int i=0;i<ntexts;++i) {
        double ratio = 1.0 - i / (float)(ntexts-1);
        double hpos = i * (0.5 * height()) / (ntexts - 1);
        string str = to_string(minVal + (maxVal - minVal) * ratio);
        painter.drawText(static_cast<int>(left + w),
                         static_cast<int>(top+hpos + 5.0), QString(str.c_str()));
      }
    }

    painter.end();
  }

  rendered_img = fbo.toImage();
  fbo.release();
}

void OffscreenBlendshapeVisualizer::enableLighting()
{
  enabled_lights.clear();

  // Setup material
  auto& mat_specular_json = rendering_settings["material"]["specular"];
  GLfloat mat_specular[] = {
    mat_specular_json[0],
    mat_specular_json[1],
    mat_specular_json[2],
    mat_specular_json[3]
  };

  auto& mat_diffuse_json = rendering_settings["material"]["diffuse"];
  GLfloat mat_diffuse[] = {
    mat_diffuse_json[0],
    mat_diffuse_json[1],
    mat_diffuse_json[2],
    mat_diffuse_json[3]
  };

  auto& mat_shininess_json = rendering_settings["material"]["shininess"];
  GLfloat mat_shininess[] = {mat_shininess_json};

  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  // Setup Lights
  auto setup_light = [&](json light_json) {
    auto light_i = GL_LIGHT0 + enabled_lights.size();

    GLfloat light_position[] = {
      light_json["pos"][0],
      light_json["pos"][1],
      light_json["pos"][2],
      light_json["pos"][3]
    };
    GLfloat light_ambient[] = {
      light_json["ambient"][0],
      light_json["ambient"][1],
      light_json["ambient"][2],
      light_json["ambient"][3]
    };
    GLfloat light_diffuse[] = {
      light_json["diffuse"][0],
      light_json["diffuse"][1],
      light_json["diffuse"][2],
      light_json["diffuse"][3]
    };
    GLfloat light_specular[] = {
      light_json["specular"][0],
      light_json["specular"][1],
      light_json["specular"][2],
      light_json["specular"][3]
    };

    glLightfv(light_i, GL_POSITION, light_position);
    glLightfv(light_i, GL_DIFFUSE, light_diffuse);
    glLightfv(light_i, GL_SPECULAR, light_specular);
    glLightfv(light_i, GL_AMBIENT, light_ambient);

    enabled_lights.push_back(light_i);
  };

  for(json::iterator it = rendering_settings["lights"].begin();
      it != rendering_settings["lights"].end();
      ++it)
  {
    setup_light(*it);
  }

  glEnable(GL_LIGHTING);
  for(auto light_i : enabled_lights) glEnable(light_i);
}

void OffscreenBlendshapeVisualizer::disableLighting()
{
  for(auto light_i : enabled_lights) glDisable(light_i);
  enabled_lights.clear();

  glDisable(GL_LIGHTING);
}

void OffscreenBlendshapeVisualizer::drawColorBar(double minVal, double maxVal) {

  // Store current state
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  // setup 2D view
  glViewport(0, 0, width(), height());
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
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

  // Restore QPainter state
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glPopAttrib();
}

void OffscreenBlendshapeVisualizer::drawMesh(const BasicMesh &m)
{
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

  GLuint image_tex;
  if(has_texture) {
    cout << "Using texture ..." << endl;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE);
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &image_tex);
    cout << image_tex << endl;
    cout << texture.width() << 'x' << texture.height() << endl;
    glBindTexture(GL_TEXTURE_2D, image_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width(), texture.height(), 0, GL_BGRA,
                 GL_UNSIGNED_BYTE, texture.bits());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  }

  // HACK disable this for rendering textured blendshapes
  glDisable(GL_CULL_FACE);

  cout << m.NumFaces() << endl;
  cout << m.NumVertices() << endl;
  cout << ao.size() << endl;

  bool use_ao = !ao.empty();
  auto set_material_with_ao = [&](float ao_value) {
    auto& mat_diffuse_json = rendering_settings["material"]["diffuse"];
    GLfloat mat_diffuse[] = {
      float(mat_diffuse_json[0]) * ao_value,
      float(mat_diffuse_json[1]) * ao_value,
      float(mat_diffuse_json[2]) * ao_value,
      float(mat_diffuse_json[3])
    };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
  };

  bool use_customized_normals = !normals.empty();

  glBegin(GL_TRIANGLES);
  for(int i=0;i<m.NumFaces();++i) {
    if(skip_faces.count(i)) continue;

    auto face_i = m.face(i);
    int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];

    // face normal
    //auto norm_i = m.normal(i);
    //glNormal3d(norm_i[0], norm_i[1], norm_i[2]);

    auto p1 = m.vertex(v1), p2 = m.vertex(v2), p3 = m.vertex(v3);
    auto tf = m.face_texture(i);

    if(use_ao) set_material_with_ao( ao[v1] );

    auto t0 = m.texture_coords(tf[0]);
    Vector3d n1;
    if(use_customized_normals) {
      n1 = Vector3d(normals[v1*3], normals[v1*3+1], normals[v1*3+2]);
      n1.normalize();
    } else {
      n1 = m.vertex_normal(v1);
    }
    glTexCoord2f(t0[0], 1-t0[1]);
    glNormal3d(n1[0], n1[1], n1[2]);
    glVertex3d(p1[0], p1[1], p1[2]);

    if(use_ao) set_material_with_ao( ao[v2] );

    auto t1 = m.texture_coords(tf[1]);
    Vector3d n2;
    if(use_customized_normals) {
      n2 = Vector3d(normals[v2*3], normals[v2*3+1], normals[v2*3+2]);
      n2.normalize();
    } else {
      n2 = m.vertex_normal(v2);
    }
    glTexCoord2f(t1[0], 1-t1[1]);
    glNormal3d(n2[0], n2[1], n2[2]);
    glVertex3d(p2[0], p2[1], p2[2]);

    if(use_ao) set_material_with_ao( ao[v3] );

    auto t2 = m.texture_coords(tf[2]);
    Vector3d n3;
    if(use_customized_normals) {
      n3 = Vector3d(normals[v3*3], normals[v3*3+1], normals[v3*3+2]);
      n3.normalize();
    } else {
      n3 = m.vertex_normal(v3);
    }
    glTexCoord2f(t2[0], 1-t2[1]);
    glNormal3d(n3[0], n3[1], n3[2]);
    glVertex3d(p3[0], p3[1], p3[2]);
  }
  glEnd();

  if(has_texture) {
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE);
  }
}

void OffscreenBlendshapeVisualizer::drawMeshWithColor(const BasicMesh &m)
{
  #if 0
  double maxVal = *(std::max_element(dists.begin(), dists.end()));
  double minVal = *(std::min_element(dists.begin(), dists.end()));
  #else
  double maxVal = error_range;
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

void OffscreenBlendshapeVisualizer::drawMeshVerticesWithColor(const BasicMesh &m)
{
  #if 0
  double maxVal = *(std::max_element(dists.begin(), dists.end()));
  double minVal = *(std::min_element(dists.begin(), dists.end()));
  #else
  double maxVal = error_range;
  double minVal = 0.0;
  #endif

  glColor4f(0.75, 0.75, 0.75, 1.0);

  glPointSize(2.0);
  glBegin(GL_POINTS);
  for(int i=0;i<m.NumVertices();++i) {
    auto vert_i = m.vertex(i);

    //auto norm_i = m.normal(i);
    //glNormal3d(norm_i[0], norm_i[1], norm_i[2]);

    auto draw_vertex = [&](double dval, decltype(vert_i) pj) {
      double hval0 = 1.0 - max(min((dval-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
      if(dval < 0) {
        hval0 = 0.0;
      }
      QColor c0 = QColor::fromHsvF(hval0*0.67, 1.0, 1.0);
      float colors0[4] = {(float)c0.redF(), (float)c0.greenF(), (float)c0.blueF(), 1.0f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors0);
      glVertex3d(pj[0], pj[1], pj[2]);
    };

    draw_vertex(dists[i], vert_i);
  }
  glEnd();
}

BlendshapeVisualizer::BlendshapeVisualizer(QWidget *parent):
  GL3DCanvas(parent)
{
  use_side_view = false;
  setSceneScale(1.5);
  setCameraPos(0, 0, 10);
  //setProjectionMode(GL3DCanvas::ORTHONGONAL);
  mouseInteractionMode = GL3DCanvas::VIEW_TRANSFORM;
}

BlendshapeVisualizer::~BlendshapeVisualizer()
{

}

void BlendshapeVisualizer::setMesh(const BasicMesh& m) {
  mesh = m;
  GL3DCanvas::repaint();
}

void BlendshapeVisualizer::loadMesh(const string &filename)
{
  mesh.LoadOBJMesh(filename);
  mesh.ComputeNormals();
  computeDistance();
  GL3DCanvas::repaint();
}

void BlendshapeVisualizer::loadReferenceMesh(const string &filename)
{
  refmesh.LoadOBJMesh(filename);
  refmesh.ComputeNormals();
  computeDistance();
  GL3DCanvas::repaint();
}

void BlendshapeVisualizer::paintGL()
{
  GL3DCanvas::paintGL();

  glEnable(GL_DEPTH_TEST);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  cout << "Enabling lighting ..." << endl;
  enableLighting();
  if( mesh.NumFaces() > 0 ) {
    if( refmesh.NumFaces() > 0 ) drawMeshWithColor(mesh);
    else drawMesh(mesh);
  } else {
    cout << "Drawing mesh ..." << endl;
    glDepthFunc(GL_LEQUAL);
    drawMesh(refmesh);
    cout << "Drawing vertices mesh ..." << endl;
    glDepthFunc(GL_ALWAYS);
    drawMeshVerticesWithColor(mesh);
  }
  disableLighting();

  if( !dists.empty() ) {
    #if 0
    double minVal = (*std::min_element(dists.begin(), dists.end()));
    double maxVal = (*std::max_element(dists.begin(), dists.end()));
    #else
    double minVal = 0.0;
    double maxVal = 0.05;
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
  boost::timer::auto_cpu_timer t("distance computation time = %w seconds.\n");
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
  #pragma omp parallel for
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

    // face normal
    //auto norm_i = m.normal(i);
    //glNormal3d(norm_i[0], norm_i[1], norm_i[2]);

    auto p1 = m.vertex(v1), p2 = m.vertex(v2), p3 = m.vertex(v3);

    auto n1 = m.vertex_normal(v1);
    glNormal3d(n1[0], n1[1], n1[2]);
    glVertex3d(p1[0], p1[1], p1[2]);

    auto n2 = m.vertex_normal(v2);
    glNormal3d(n2[0], n2[1], n2[2]);
    glVertex3d(p2[0], p2[1], p2[2]);

    auto n3 = m.vertex_normal(v3);
    glNormal3d(n3[0], n3[1], n3[2]);
    glVertex3d(p3[0], p3[1], p3[2]);
  }
  glEnd();
}

void BlendshapeVisualizer::drawMeshVerticesWithColor(const BasicMesh &m)
{
  #if 0
  double maxVal = *(std::max_element(dists.begin(), dists.end()));
  double minVal = *(std::min_element(dists.begin(), dists.end()));
  #else
  double maxVal = 0.05;
  double minVal = 0.0;
  #endif

  glColor4f(0.75, 0.75, 0.75, 1.0);
  glBegin(GL_POINTS);
  for(int i=0;i<m.NumVertices();++i) {
    auto vert_i = m.vertex(i);

    //auto norm_i = m.normal(i);
    //glNormal3d(norm_i[0], norm_i[1], norm_i[2]);

    auto draw_vertex = [&](double dval, decltype(vert_i) pj) {
      double hval0 = 1.0 - max(min((dval-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
      if(dval < 0) {
        hval0 = 0.0;
      }
      QColor c0 = QColor::fromHsvF(hval0*0.67, 1.0, 1.0);
      float colors0[4] = {(float)c0.redF(), (float)c0.greenF(), (float)c0.blueF(), 1.0f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, colors0);
      //glNormal3d(nj[0], nj[1], nj[2]);
      glVertex3d(pj[0], pj[1], pj[2]);
    };

    draw_vertex(dists[i], vert_i);

  }
  glEnd();
}

void BlendshapeVisualizer::drawMeshWithColor(const BasicMesh &m)
{
  #if 0
  double maxVal = *(std::max_element(dists.begin(), dists.end()));
  double minVal = *(std::min_element(dists.begin(), dists.end()));
  #else
  double maxVal = 0.05;
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

BlendshapeGeneration::BlendshapeGeneration(bool silent, QWidget *parent)
    : QMainWindow(parent), silent(silent)
{
    ui.setupUi(this);

    if (silent) {
      ocanvas = new OffscreenBlendshapeVisualizer(512, 512);
    } else {
      canvas = new BlendshapeVisualizer(this);
      setCentralWidget(canvas);
      connect(ui.actionLoad_Mesh, SIGNAL(triggered()), this, SLOT(slot_loadMesh()));
      connect(ui.actionLoad_Reference, SIGNAL(triggered()), this, SLOT(slot_loadReferenceMesh()));
    }
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
