#ifndef INTERACTIVE_RIGGING_H
#define INTERACTIVE_RIGGING_H

#include <QtWidgets/QMainWindow>

#include "OpenGL/gl3dcanvas.h"
#include <MultilinearReconstruction/basicmesh.h>

#include "blendshapegeneration.h"
#include "blendshapesweightswidget.h"

class InteractiveRigging : public QMainWindow {
  Q_OBJECT
public:
  InteractiveRigging(QWidget *parent=0);
  ~InteractiveRigging();

  virtual QSize sizeHint() const {
    return QSize(720, 720);
  }

  virtual QSize minimumSizeHint() const {
    return QSize(720, 720);
  }

  void LoadBlendshapes(const string& path);

  virtual void keyPressEvent(QKeyEvent* e);

private slots:
  void slot_weightsChanged(int, int);

private:
  void UpdateShape();


private:
  vector<BasicMesh> blendshapes;

  BlendshapeVisualizer *canvas;
  BlendshapesWeightsWidget weights_widget;

  BasicMesh mesh;
  vector<Eigen::MatrixX3d> vertices;
  vector<double> weights;

  int cur_idx;
};

#endif // INTERACTIVE_RIGGING_H
