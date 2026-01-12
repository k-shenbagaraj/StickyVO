#pragma once
#include "types.hpp"
#include "map.hpp"
#include "pnp.hpp"

namespace stickyvo {

struct LocalRefineParams {
  int max_landmarks = 500; // Cap landmarks for performance
};

/**
 * Perform a PnP refinement using a subset of the local map.
 * Typically called after an initial pose estimate is obtained.
 */
PnPResult refine_pose_locally(
    const CameraIntrinsics& K,
    const Map& map,
    const std::vector<FeatureObs>& obs,
    const std::vector<LineObs>& line_obs,
    const PnPParams& pp,
    const LocalRefineParams& rp,
    const std::optional<Pose>& init);

} // namespace stickyvo
