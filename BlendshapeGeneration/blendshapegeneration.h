#ifndef BLENDSHAPEGENERATION_H
#define BLENDSHAPEGENERATION_H

#include <QtWidgets/QMainWindow>
#include "ui_blendshapegeneration.h"

#include "OpenGL/gl3dcanvas.h"
#include <MultilinearReconstruction/basicmesh.h>

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

  void loadMesh(const string &filename);
  void loadReferenceMesh(const string &filename);

protected:
  virtual void paintGL();

  void enableLighting();
  void disableLighting();
  void computeDistance();
  void drawMesh(const BasicMesh &m);
  void drawMeshWithColor(const BasicMesh &m);
  void drawColorBar(double, double);

private:
  vector<double> dists;
  BasicMesh mesh;
  BasicMesh refmesh;
};

class BlendshapeGeneration : public QMainWindow
{
  Q_OBJECT

public:
  BlendshapeGeneration(QWidget *parent = 0);
  ~BlendshapeGeneration();

  virtual QSize sizeHint() const {
    return QSize(600, 600);
  }

  void LoadMeshes(const string& mesh, const string& refmesh) {
    canvas->loadMesh(mesh);
    canvas->loadReferenceMesh(refmesh);
    canvas->repaint();
  }

  void LoadMesh(const string& mesh) {
    canvas->loadMesh(mesh);
    canvas->repaint();
  }

  void Save(const string& filename) {
    QImage pixmap = canvas->grabFrameBuffer();
    pixmap.save(filename.c_str());
  }

private slots:
  void slot_loadMesh();
  void slot_loadReferenceMesh();

private:
  Ui::BlendshapeGenerationClass ui;

  BlendshapeVisualizer *canvas;
};

#endif // BLENDSHAPEGENERATION_H
