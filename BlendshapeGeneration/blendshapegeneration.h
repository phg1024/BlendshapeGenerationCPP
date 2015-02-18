#ifndef BLENDSHAPEGENERATION_H
#define BLENDSHAPEGENERATION_H

#include <QtWidgets/QMainWindow>
#include "ui_blendshapegeneration.h"

class BlendshapeGeneration : public QMainWindow
{
    Q_OBJECT

public:
    BlendshapeGeneration(QWidget *parent = 0);
    ~BlendshapeGeneration();

private:
    Ui::BlendshapeGenerationClass ui;
};

#endif // BLENDSHAPEGENERATION_H
