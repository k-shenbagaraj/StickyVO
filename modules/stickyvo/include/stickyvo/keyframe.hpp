#pragma once

#include "types.hpp"

namespace stickyvo {

/**
 * Keyframe stores a snapshot of the camera pose and all observations (point/line)
 * captured at a specific point in the trajectory.
 */
struct Keyframe {
  FrameId frame_id = 0;
  double t_sec = 0.0;
  Pose T_wc;
  std::vector<FeatureObs> obs;
  std::vector<LineObs> line_obs;
};

} // namespace stickyvo
