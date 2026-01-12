#pragma once
#include <cstdint>
#include <vector>

namespace stickyvo_lgs {

struct Vec2f { float x{0}, y{0}; };

struct Line2D {
  Vec2f p0;
  Vec2f p1;
};

struct Match { int i0{-1}, i1{-1}; };

// --- Feature Collections ---

struct FrameFeatures {
  std::vector<Vec2f> keypoints;
  std::vector<Line2D> lines;
};

struct PairMatches {
  FrameFeatures f0;
  FrameFeatures f1;
  std::vector<Match> point_matches;
  std::vector<Match> line_matches;

  int num_inliers_points{0};
  int num_inliers_lines{0};
  double score{0.0};
};

// --- View Definitions ---

struct CameraIntrinsics {
  double fx{0}, fy{0}, cx{0}, cy{0};
};

struct ImageView {
  const uint8_t* data{nullptr};
  int width{0};
  int height{0};
  int stride_bytes{0};

  enum class Format { kGray8, kRGB8, kBGR8 } format{Format::kGray8};
};

} // namespace stickyvo_lgs
