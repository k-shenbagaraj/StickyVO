#pragma once

#include "types.hpp"
#include <opencv2/core.hpp>

namespace stickyvo {

struct TriangulationParams {
  double min_parallax_deg = 1.0;
  double min_depth = 0.1;
  double max_depth = 1000.0;
  double max_reproj_err_px = 3.0; // Max allowed pixel error for valid landmark
};

struct TriangulationResult {
  bool ok = false;
  Vec3 p_w = Vec3::Zero();
  double parallax_deg = 0.0;
  double reproj_err_px = 0.0;
};

struct TriangulateLineResult {
  bool ok = false;
  Line3D line_w;
  double reproj_err_px = 0.0;
};

// Reconstruct 3D point from two-view correspondences
TriangulationResult triangulate_two_view(
    const Pose& T_wc0, const Pose& T_wc1,
    const CameraIntrinsics& K,
    const Vec2& px0, const Vec2& px1,
    const TriangulationParams& tp);

// Reconstruct 3D line from two-view line observations
TriangulateLineResult triangulate_line_two_view(
    const Pose& T_wc0, const Pose& T_wc1,
    const CameraIntrinsics& K,
    const LineObs& lo0, const LineObs& lo1,
    const TriangulationParams& tp);

// Calculate angular parallax between two specific pixel observations
double parallax_deg(
    const Pose& T_wc0, const Pose& T_wc1,
    const CameraIntrinsics& K,
    const Vec2& px0, const Vec2& px1);

} // namespace stickyvo
