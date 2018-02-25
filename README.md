# BlendshapeGenerationCPP

## Dependencies
* Boost 1.63
* ceres solver 1.12.0
* OpenCV 3.4
* freeglut
* GLEW
* Eigen 3.3.3
* SuiteSparse 4.5.3
* Intel MKL
* Qt5
* PhGLib

## Compile
```bash
git clone --recursive https://github.com/phg1024/BlendshapeGenerationCPP.git
cd BlendshapeGenerationCPP/BlendshapeGeneration
mkdir build
cd build
cmake .. -DCMAKE_BUILT_TYPE=Release -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
make -j8
```
