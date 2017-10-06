#ifndef BLENDSHAPEGENERATION_H
#define BLENDSHAPEGENERATION_H

#include <QtWidgets/QMainWindow>
#include "ui_blendshapegeneration.h"

#include "OpenGL/gl3dcanvas.h"
#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/ioutilities.h>

#include <boost/timer/timer.hpp>

#include "json/src/json.hpp"
using json = nlohmann::json;

class OffscreenBlendshapeVisualizer {
public:
  OffscreenBlendshapeVisualizer(int w, int h);
  ~OffscreenBlendshapeVisualizer();

  void setRenderingSettings(json settings) {
    rendering_settings = settings;
    SetBasicRenderingParams();
  }

  void SetSideView(bool val) {
    use_side_view = val;
  }

  void SetAmbientOcclusion(const vector<float>& AO) {
    ao = AO;
  }

  void SetSkipFaces(const vector<int>& skip_faces_in, bool subdivided=false) {
    vector<int> faces_to_skip;
    for(auto i : skip_faces_in) {
      faces_to_skip.push_back(2*i);
      faces_to_skip.push_back(2*i+1);
    }
    if(subdivided) {
      //cout << "subdivided" << endl;
      for(auto i : faces_to_skip) {
        skip_faces.insert(4*i);
        skip_faces.insert(4*i+1);
        skip_faces.insert(4*i+2);
        skip_faces.insert(4*i+3);
      }
    } else {
      skip_faces = set<int>(faces_to_skip.begin(), faces_to_skip.end());
    }
  }

  void loadMesh(const string& filename);
  void loadReferenceMesh(const string& filename);

  void setTexture(const QImage& img) {
    texture = img;
    has_texture = true;
  }

  double getError() const {
    boost::timer::auto_cpu_timer t("[compute error] error computation time = %w seconds.\n");
    double avg_error = 0;
    int cnt = 0;
    for(auto x : dists) {
      if(x >= 0) {
        ++cnt;
        avg_error += x;
      }
    }
    return avg_error / cnt;
  }

  int width() const { return canvasW; }
  int height() const { return canvasH; }

  QImage render() {
    paint();
    return rendered_img;
  }

  QImage getImage() {
    return rendered_img;
  }

  void repaint() {

  }

protected:
  void SetBasicRenderingParams();
  void setupViewing();
  void paint();
  void enableLighting();
  void disableLighting();
  void computeDistance();
  void drawMesh(const BasicMesh& m);
  void drawMeshWithColor(const BasicMesh& m);
  void drawMeshVerticesWithColor(const BasicMesh& m);
  void drawColorBar(double, double);

private:
  bool use_side_view;
  vector<double> dists;
  vector<float> ao;
  BasicMesh mesh;
  BasicMesh refmesh;

  set<int> skip_faces;

  double sceneScale;
  QVector3D cameraPos;

  int canvasW, canvasH;
  enum ProjectionMode {
    ORTHONGONAL,
    FRUSTUM,
    PERSPECTIVE
  } projectionMode;
  QImage rendered_img;

  bool has_texture;
  QImage texture;

  json rendering_settings;
  vector<GLuint> enabled_lights;
};

class BlendshapeVisualizer : public GL3DCanvas {
  Q_OBJECT
public:
  BlendshapeVisualizer(QWidget *parent = 0);
  ~BlendshapeVisualizer();

  virtual QSize sizeHint() const {
    return QSize(600, 600);
  }

  virtual QSize minimumSizeHint() const {
    return QSize(600, 600);
  }

  void SetAmbientOcclusion(const vector<float>& AO) {
    ao = AO;
  }
  void SetSideView(bool val) {
    use_side_view = val;
  }
  void setMesh(const BasicMesh& m);
  void loadMesh(const string &filename);
  void loadReferenceMesh(const string &filename);

  QImage getImage() {
    return this->grabFrameBuffer();
  }

  QImage render() {
    repaint();
    return this->grabFrameBuffer();
  }

protected:
  virtual void paintGL();

  void enableLighting();
  void disableLighting();
  void computeDistance();
  void drawMesh(const BasicMesh &m);
  void drawMeshWithColor(const BasicMesh &m);
  void drawMeshVerticesWithColor(const BasicMesh& m);
  void drawColorBar(double, double);

private:
  bool use_side_view;
  vector<double> dists;
  vector<float> ao;
  BasicMesh mesh;
  BasicMesh refmesh;
};

class BlendshapeGeneration : public QMainWindow
{
  Q_OBJECT

public:
  BlendshapeGeneration(bool silent, QWidget *parent = 0);
  ~BlendshapeGeneration();

  virtual QSize sizeHint() const {
    return QSize(600, 600);
  }

  void SetSideView(bool val) {
    if(silent) {
      ocanvas->SetSideView(val);
    } else {
      canvas->SetSideView(val);
    }
  }

  void SetTexture(const string& filename) {
    texture = QImage(filename.c_str());
    if(silent) {
      ocanvas->setTexture(texture);
    }
  }

  void SetAmbientOcclusion(const string& filename) {
    vector<float> ao = LoadFloats(filename);
    if(silent) {
      ocanvas->SetAmbientOcclusion(ao);
    } else {
      canvas->SetAmbientOcclusion(ao);
    }
  }

  void LoadRenderingSettings(const string& filename) {
    json settings;
    ifstream fin(filename);
    fin >> settings;

    if(silent) {
      ocanvas->setRenderingSettings(settings);
    }
  }

  void LoadMeshes(const string& mesh, const string& refmesh) {
    if(silent) {
      ocanvas->loadMesh(mesh);
      ocanvas->loadReferenceMesh(refmesh);
      avg_error = ocanvas->getError();
    } else {
      canvas->loadMesh(mesh);
      canvas->loadReferenceMesh(refmesh);
      canvas->repaint();
      //avg_error = canvas->getError();
    }
  }

  void LoadMesh(const string& mesh) {
    if(silent) {
      ocanvas->loadMesh(mesh);
    } else {
      canvas->loadMesh(mesh);
      canvas->repaint();
    }
  }

  void LoadSkipFaces(const string& filename, bool subdivided) {
    auto indices = LoadIndices(filename);
    if(silent) {
      ocanvas->SetSkipFaces(indices, subdivided);
    }
  }

  void Save(const string& filename) {
    boost::timer::auto_cpu_timer t("[save image] image save time = %w seconds.\n");
    if(silent) {
      QImage pixmap = ocanvas->render();
      pixmap.save(filename.c_str());
    } else {
      QImage pixmap = canvas->render();
      pixmap.save(filename.c_str());
    }
  }

  void SaveError(const string& filename) {
    boost::timer::auto_cpu_timer t("[save error] error save time = %w seconds.\n");
    ofstream fout(filename);
    fout << avg_error << endl;
    fout.close();
  }

private slots:
  void slot_loadMesh();
  void slot_loadReferenceMesh();

private:
  Ui::BlendshapeGenerationClass ui;

  bool silent;
  double avg_error;
  BlendshapeVisualizer *canvas;
  OffscreenBlendshapeVisualizer *ocanvas;

  QImage texture;
};

#endif // BLENDSHAPEGENERATION_H
