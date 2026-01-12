#include "stickyvo/local_refine.hpp"

namespace stickyvo {

PnPResult refine_pose_locally(const CameraIntrinsics& K, const Map& map, const std::vector<FeatureObs>& obs,
                              const std::vector<LineObs>& line_obs, const PnPParams& pp,
                              const LocalRefineParams& rp, const std::optional<Pose>& init) {
  std::vector<Landmark> lms;
  lms.reserve(std::min((int)map.lms.size(), rp.max_landmarks));
  
  // Select a subset of active landmarks for multi-frame refinement
  for (const auto& kv : map.lms) {
    if (!kv.second.valid) continue;
    lms.push_back(kv.second);
    if ((int)lms.size() >= rp.max_landmarks) break;
  }
  
  return estimate_pose_pnp(K, lms, obs, line_obs, pp, init);
}

} // namespace stickyvo
