#pragma once

#include "types.hpp"
#include "map.hpp"

namespace stickyvo {

struct BundleAdjustParams {
  int window_keyframes = 4;
  int max_iterations = 15;
  double huber_delta_px = 3.0;
  double max_cost_increase = 1.5; // Rejects BA result if cost explodes
  double max_motion_thresh = 1.0;  // Rejects BA result if pose drifts too far
};

/**
 * Perform sliding-window bundle adjustment on a subset of keyframes.
 * Optimizes both camera poses and landmark positions.
 */
bool run_local_bundle_adjustment(const CameraIntrinsics& K,
                                Map& map,
                                const BundleAdjustParams& bp);

} // namespace stickyvo
