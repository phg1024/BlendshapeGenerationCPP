#include "blendshapesweightswidget.h"
#include "ui_blendshapesweightswidget.h"

#include <QSlider>
#include <QLabel>

BlendshapesWeightsWidget::BlendshapesWeightsWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::BlendshapesWeightsWidget)
{
    ui->setupUi(this);

    setLayout(ui->gridLayout);

    for(int i=0;i<46;++i) {
        auto* slider = new QSlider(Qt::Horizontal);
        sliders.push_back(slider);
        int r = i % 23;
        int c = (i / 23) * 2 + 1;
        ui->gridLayout->addWidget(new QLabel(QString::fromStdString(to_string(i+1))), r, c-1);
        ui->gridLayout->addWidget(slider, r, c);

        connect(slider, SIGNAL(valueChanged(int)), this, SLOT(slot_sliderChanged(int)));
    }
}

BlendshapesWeightsWidget::~BlendshapesWeightsWidget()
{
    delete ui;
}

void BlendshapesWeightsWidget::slot_sliderChanged(int v)
{
    auto s = dynamic_cast<QSlider*>(sender());
    for(int i=0;i<sliders.size();++i) {
        if( s == sliders[i] ) {
            emit sig_sliderChanged(i + 1, v);
        }
    }
}
