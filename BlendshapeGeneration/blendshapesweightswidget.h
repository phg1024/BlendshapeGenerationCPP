#ifndef BLENDSHAPESWEIGHTSWIDGET_H
#define BLENDSHAPESWEIGHTSWIDGET_H

#include "ui_blendshapesweightswidget.h"

#include <QWidget>
#include <QSlider>
#include <vector>
using namespace std;

namespace Ui {
class BlendshapesWeightsWidget;
}

class BlendshapesWeightsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit BlendshapesWeightsWidget(QWidget *parent = 0);
    ~BlendshapesWeightsWidget();

signals:
    void sig_sliderChanged(int, int);

private slots:
    void slot_sliderChanged(int);

private:
    Ui::BlendshapesWeightsWidget *ui;

    vector<QSlider*> sliders;
};

#endif // BLENDSHAPESWEIGHTSWIDGET_H
