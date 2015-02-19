QT += core widgets
HEADERS += blendshapegeneration.h \
    BasicMatrix.hpp \
    BasicMesh.h \
    common.h \
    MeshDeformer.h \
    PointCloud.h \
    SparseMatrix.hpp
SOURCES += blendshapegeneration.cpp \
           main.cpp \
    BasicMesh.cpp \
    MeshDeformer.cpp \
    PointCloud.cpp
RESOURCES += blendshapegeneration.qrc

FORMS += \
    blendshapegeneration.ui

CONFIG += c++11
CONFIG -= app_bundle

INCLUDEPATH += /usr/local/include /opt/intel/mkl/include
LIBS += -L/opt/intel/lib -L/opt/intel/mkl/lib -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm

INCLUDEPATH += /Users/phg/SDKs/PhGLib/include
LIBS += -L/Users/phg/SDKs/PhGLib/lib -lPhGLib

INCLUDEPATH += /Users/phg/SDKs/SuiteSparse/SuiteSparse_config /Users/phg/SDKs/SuiteSparse/CHOLMOD/Include
LIBS += -L/Users/phg/SDKs/SuiteSparse/SuiteSparse_config -L/Users/phg/SDKs/SuiteSparse/AMD/Lib -L/Users/phg/SDKs/SuiteSparse/COLAMD/Lib -L/Users/phg/SDKs/SuiteSparse/CHOLMOD/Lib -lcholmod -lamd -lcolamd -lsuitesparseconfig
