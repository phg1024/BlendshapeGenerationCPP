QT += core widgets
HEADERS += blendshapegeneration.h \
    BasicMesh.h \
    common.h \
    MeshDeformer.h \
    PointCloud.h \
    sparsematrix.h \
    densematrix.h \
    densevector.h \
    ndarray.hpp
SOURCES += blendshapegeneration.cpp \
           main.cpp \
    BasicMesh.cpp \
    MeshDeformer.cpp \
    PointCloud.cpp \
    common.cpp \
    sparsematrix.cpp \
    densematrix.cpp \
    densevector.cpp
RESOURCES += blendshapegeneration.qrc

FORMS += \
    blendshapegeneration.ui

CONFIG += c++11
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -m64

INCLUDEPATH += /usr/local/include /opt/intel/mkl/include
LIBS += -L/opt/intel/lib -L/opt/intel/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm

INCLUDEPATH += /Users/phg/SDKs/PhGLib/include
LIBS += -L/Users/phg/SDKs/PhGLib/lib -lPhGLib

INCLUDEPATH += /Users/phg/SDKs/SuiteSparse/SuiteSparse_config /Users/phg/SDKs/SuiteSparse/CHOLMOD/Include
LIBS += -L/Users/phg/SDKs/SuiteSparse/SuiteSparse_config -L/Users/phg/SDKs/SuiteSparse/AMD/Lib -L/Users/phg/SDKs/SuiteSparse/COLAMD/Lib -L/Users/phg/SDKs/SuiteSparse/CHOLMOD/Lib -lcholmod -lamd -lcolamd -lsuitesparseconfig


LIBS += -framework Accelerate
