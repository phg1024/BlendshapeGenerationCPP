#include "interactiverigging.h"

#include <QFileDialog>
#include <QImage>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLPaintDevice>

InteractiveRigging::InteractiveRigging(QWidget* parent)
  : QMainWindow(parent) {
  canvas = new BlendshapeVisualizer(this);

  cur_idx = 0;

  setCentralWidget(canvas);

  weights_widget.show();

  connect(&weights_widget, SIGNAL(sig_sliderChanged(int, int)), this, SLOT(slot_weightsChanged(int,int)));
}

InteractiveRigging::~InteractiveRigging() {}

void InteractiveRigging::slot_weightsChanged(int widx, int tick) {
    weights[widx] = tick * 0.01;
    UpdateShape();
    canvas->setMesh(mesh);
}

void InteractiveRigging::UpdateShape() {
    MatrixX3d v = vertices[0];
    for(int i=1;i<47;++i) {
        if(weights[i] > 0) v += vertices[i] * weights[i];
    }
    mesh.vertices() = v;
    //mesh.ComputeNormals();
}

void InteractiveRigging::LoadBlendshapes(const string& path) {
  const string prefix = "B_";
  const int num_blendshapes = 47;
  blendshapes.resize(num_blendshapes);
  for(int i=0;i<num_blendshapes;++i) {
    blendshapes[i].LoadOBJMesh(path + "/" + prefix + to_string(i) + ".obj");
    blendshapes[i].ComputeNormals();
  }

  weights.resize(num_blendshapes, 0);
  weights[0] = 1.0;

  vertices.resize(num_blendshapes);
  vertices[0] = blendshapes.front().vertices();
  for(int i=1;i<num_blendshapes;++i) {
      vertices[i] = blendshapes[i].vertices() - vertices[0];
  }

  mesh = blendshapes.front();

  canvas->setMesh(mesh);

  grabKeyboard();  
}

void InteractiveRigging::keyPressEvent(QKeyEvent* e) {
  switch (e->key()) {
    case Qt::Key_Left: {
      cur_idx = (cur_idx + 1) % blendshapes.size();
      mesh = blendshapes[cur_idx];
      canvas->setMesh(mesh);
      break;
    }
    case Qt::Key_Right: {
      cur_idx = (cur_idx - 1);
      cur_idx = (cur_idx < 0)?cur_idx+blendshapes.size():cur_idx;
      mesh = blendshapes[cur_idx];
      canvas->setMesh(mesh);
      break;
    }
  }
}
