QT += core widgets
HEADERS += blendshapegeneration.h \
    common.h \
    sparsematrix.h \
    densematrix.h \
    densevector.h \
    ndarray.hpp \
    sparsematrix.h \
    cereswrapper.h \
    meshtransferer.h \
    meshdeformer.h \
    pointcloud.h \
    basicmesh.h
SOURCES += blendshapegeneration.cpp \
           main.cpp \
    common.cpp \
    sparsematrix.cpp \
    densematrix.cpp \
    densevector.cpp \
    meshtransferer.cpp \
    meshdeformer.cpp \
    pointcloud.cpp \
    basicmesh.cpp
RESOURCES += blendshapegeneration.qrc

FORMS += \
    blendshapegeneration.ui

CONFIG += c++11
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -m64 -frounding-math

macx {
message(building for mac os)
INCLUDEPATH += /usr/local/include /opt/intel/mkl/include
LIBS += -L/opt/intel/lib -L/opt/intel/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm

INCLUDEPATH += /Users/phg/SDKs/PhGLib/include
LIBS += -L/Users/phg/SDKs/PhGLib/lib -lPhGLib

INCLUDEPATH += /Users/phg/SDKs/SuiteSparse/SuiteSparse_config /Users/phg/SDKs/SuiteSparse/CHOLMOD/Include
LIBS += -L/Users/phg/SDKs/SuiteSparse/SuiteSparse_config -L/Users/phg/SDKs/SuiteSparse/AMD/Lib -L/Users/phg/SDKs/SuiteSparse/COLAMD/Lib -L/Users/phg/SDKs/SuiteSparse/CHOLMOD/Lib -lcholmod -lamd -lcolamd -lsuitesparseconfig

LIBS += -framework Accelerate
}

linux-g++ {
message(building for linux)

QMAKE_CXXFLAGS += -fopenmp

# mkl
INCLUDEPATH += /usr/local/include /opt/intel/mkl/include
LIBS += -L/opt/intel/lib/intel64/ -L/opt/intel/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread -lm

# PhGLib
INCLUDEPATH += /home/phg/SDKs/PhGLib/include
LIBS += -L/home/phg/SDKs/PhGLib/lib -lPhGLib

# ceres
INCLUDEPATH += /home/phg/SDKs/ceres-solver-1.10.0/include /usr/include/eigen3
LIBS += -L/home/phg/SDKs/ceres-solver-1.10.0/lib -lceres -lglog -gflags

# CHOLMOD
#INCLUDEPATH += /home/phg/SDKs/SuiteSparse/SuiteSparse_config /home/phg/SDKs/SuiteSparse/CHOLMOD/Include
#LIBS += -L/home/phg/SDKs/SuiteSparse/SuiteSparse_config -L/home/phg/SDKs/SuiteSparse/AMD/Lib -L/home/phg/SDKs/SuiteSparse/COLAMD/Lib -L/home/phg/SDKs/SuiteSparse/CHOLMOD/Lib -lcholmod -lamd -lcolamd -lsuitesparseconfig
LIBS += -lcholmod -lcamd -lamd -lccolamd -lcolamd -lsuitesparseconfig -lcxsparse

# CGAL
LIBS += -lCGAL -lboost_system
}

