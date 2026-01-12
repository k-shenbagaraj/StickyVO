#pragma once

#include "types.hpp"
#include <opencv2/core.hpp>

namespace stickyvo {

struct PnPParams {
  int min_inliers = 30;
  double ransac_reproj_thresh_px = 5.0;
  int ransac_iters = 200;
  double ransac_conf = 0.999;
};

struct PnPResult {
  bool ok = false;
  Pose T_wc;
  int inliers = 0;
};

// Robustly estimate camera pose from 3D-2D correspondences
PnPResult estimate_pose_pnp(
    const CameraIntrinsics& K,
    const std::vector<Landmark>& lms,
    const std::vector<FeatureObs>& obs,
    const std::vector<LineObs>& line_obs,
    const PnPParams& pp,
    const std::optional<Pose>& init = std::nullopt);

// Refine pose using non-linear optimization (PnPL: Points and Lines)
PnPResult refine_pose_pnpl(
    const CameraIntrinsics& K,
    const std::vector<Landmark>& lms,
    const std::vector<FeatureObs>& obs,
    const std::vector<LineObs>& line_obs,
    const Pose& init_pose);

} // namespace stickyvo
