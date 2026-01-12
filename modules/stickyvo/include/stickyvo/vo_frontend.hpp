#pragma once

#include "types.hpp"
#include "map.hpp"
#include "triangulation.hpp"
#include "pnp.hpp"
#include "local_refine.hpp"
#include "bundle_adjustment.hpp"

namespace stickyvo {

struct FrontendParams {
  // Keyframe logic
  int min_tracked_features = 150;
  double min_keyframe_parallax_deg = 0.5;
  int max_frames_between_keyframes = 8;

  TriangulationParams tri;
  PnPParams pnp;
  LocalRefineParams refine;

  // Landmark lifecycle
  int map_max_keyframes = 10;
  int min_obs_for_landmark = 2;

  double prune_reproj_thresh_px = 6.0;
  int prune_bad_streak_thresh = 3;
  int prune_min_obs = 3;
  int prune_grace_frames = 15;

  // Local optimization
  bool enable_local_ba = true;
  int ba_window_keyframes = 4;
  int ba_max_iterations = 20;
  double ba_max_cost_increase = 1.5;
  double ba_max_motion_thresh = 1.0;
  
  double pnp_max_motion_thresh = 20.0;
};

struct FrontendOutput {
  bool pose_ok = false;
  Pose T_wc;
  bool made_keyframe = false;
  int pnp_inliers = 0;
  int num_landmarks = 0;
};

/**
 * VoFrontend handles frame-to-frame pose estimation, keyframe promotion, 
 * triangulation of new landmarks, and local bundle adjustment.
 */
class VoFrontend {
public:
  explicit VoFrontend(const FrontendParams& p): p_(p) {
    map_.max_keyframes = p_.map_max_keyframes;
  }

  void reset();

  FrontendOutput process_frame(
      FrameId frame_id, double t_sec,
      const CameraIntrinsics& K,
      const std::vector<FeatureObs>& obs,
      const std::vector<LineObs>& line_obs,
      const std::optional<Quat>& q_wc_prior = std::nullopt);

  const Map& map() const { return map_; }
  const std::optional<Keyframe>& last_keyframe() const { return last_kf_; }

private:
  FrontendParams p_;
  Map map_;

  std::optional<Pose> last_pose_;
  std::optional<Pose> prev_pose_;
  std::optional<Keyframe> last_kf_;
  int frames_since_kf_ = 0;

  // --- Internal Pipeline Steps ---
  bool should_make_keyframe(const CameraIntrinsics& K,
                            const std::vector<FeatureObs>& obs,
                            const std::vector<LineObs>& line_obs,
                            const Pose& T_wc_est) const;

  void create_keyframe(FrameId frame_id, double t_sec,
                       const Pose& T_wc,
                       const std::vector<FeatureObs>& obs,
                       const std::vector<LineObs>& line_obs);

  void triangulate_with_last_keyframe(const CameraIntrinsics& K,
                                      const Keyframe& kf0,
                                      const Keyframe& kf1);

  std::vector<Landmark> collect_local_landmarks() const;
};

} // namespace stickyvo
