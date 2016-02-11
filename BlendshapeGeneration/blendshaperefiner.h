#ifndef FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
#define FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H

class BlendshapeRefiner {
public:
  BlendshapeRefiner() {}
  ~BlendshapeRefiner() {}

  void SetBlendshapeCount(int count) { num_shapes = count; }
  void LoadTemplateMeshes(const string& path, const string& basename);
  void LoadInputReconstructionResults(const string& settings_filename);
  void LoadInputPointClouds(const string& path);

  void Refine();

private:
  void CreateTrainingShapes();
  void InitializeBlendshapes();

private:
  int num_shapes;
  vector<BasicMesh> A;      // template blendshapes

  vector<BasicMesh> Binit;  // transferred blendshapes
  vector<BasicMesh> B;      // refined blendshapes

  int num_poses;
  vector<BasicMesh> S0;     // initial training shapes
  vector<BasicMesh> S;      // point cloud deformed training shapes
};

#endif //FACESHAPEFROMSHADING_BLENDSHAPEREFINER_H
