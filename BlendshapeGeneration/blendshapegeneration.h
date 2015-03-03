#ifndef BLENDSHAPEGENERATION_H
#define BLENDSHAPEGENERATION_H

#include <QtWidgets/QMainWindow>
#include "ui_blendshapegeneration.h"

#include "OpenGL/gl3dcanvas.h"
#include "basicmesh.h"

class BlendshapeVisualizer : public GL3DCanvas {
  Q_OBJECT
public:
  BlendshapeVisualizer(QWidget *parent = 0);
  ~BlendshapeVisualizer();

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

private slots:
  void slot_loadMesh();
  void slot_loadReferenceMesh();

private:
  Ui::BlendshapeGenerationClass ui;

  BlendshapeVisualizer *canvas;
};

#endif // BLENDSHAPEGENERATION_H
