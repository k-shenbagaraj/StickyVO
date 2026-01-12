#include "stickyvo/vo_frontend.hpp"
#include <unordered_map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <cstdio>

namespace stickyvo {

// --- Lifecycle ---

void VoFrontend::reset() {
  map_ = Map{};
  map_.max_keyframes = p_.map_max_keyframes;
  last_pose_.reset();
  prev_pose_.reset();
  last_kf_.reset();
  frames_since_kf_ = 0;
}

// --- Pose Utilities ---

static Eigen::Isometry3d pose_to_T(const Pose& T_wc) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = T_wc.q_wc.normalized().toRotationMatrix();
  T.translation() = T_wc.p_wc;
  return T;
}

static Pose T_to_pose(const Eigen::Isometry3d& T) {
  Pose out;
  out.p_wc = T.translation();
  out.q_wc = Quat(T.linear());
  out.q_wc.normalize();
  return out;
}

static Pose compose_pose(const Pose& A_wc, const Pose& B_wc) {
  return T_to_pose(pose_to_T(A_wc) * pose_to_T(B_wc));
}

static Pose inverse_pose(const Pose& T_wc) {
  return T_to_pose(pose_to_T(T_wc).inverse());
}

static Pose constant_velocity_predict(const Pose& prev_wc, const Pose& last_wc) {
  Pose A = compose_pose(last_wc, inverse_pose(prev_wc));
  return compose_pose(A, last_wc);
}

static inline Vec2 project_px(const CameraIntrinsics& K, const Pose& T_wc, const Vec3& X_w) {
  Vec3 X_c = T_wc.R_cw() * (X_w - T_wc.p_wc);
  const double z = X_c.z();
  if (z <= 1e-9) return Vec2(1e9, 1e9);
  return Vec2(K.fx * (X_c.x() / z) + K.cx, K.fy * (X_c.y() / z) + K.cy);
}

// --- Map Management ---

static void update_landmark_stats_and_prune(
    Map& map, FrameId frame_id, const CameraIntrinsics& K, const Pose& T_wc,
    const std::vector<FeatureObs>& obs, const FrontendParams& p)
{
  for (const auto& o : obs) {
    auto it = map.lms.find(o.id);
    if (it == map.lms.end()) continue;
    Landmark& lm = it->second;
    if (!lm.valid) continue;

    const double err = (project_px(K, T_wc, lm.p_w) - o.px).norm();
    lm.last_reproj_err_px = err;
    
    if (err <= p.prune_reproj_thresh_px) {
      lm.bad_reproj_streak = 0;
      lm.num_obs += 1;
      lm.last_seen_frame = frame_id;
    } else {
      lm.bad_reproj_streak += 1;
    }
  }

  // Prune bad landmarks
  for (auto it = map.lms.begin(); it != map.lms.end(); ) {
    Landmark& lm = it->second;
    const bool bad_streak = (lm.bad_reproj_streak >= p.prune_bad_streak_thresh);
    const bool too_few_obs = ((int)lm.num_obs < p.prune_min_obs) &&
                             (frame_id > lm.first_seen_frame + (FrameId)p.prune_grace_frames);
    
    if (!lm.valid || bad_streak || too_few_obs) it = map.lms.erase(it);
    else ++it;
  }
}

static std::unordered_map<TrackId, FeatureObs> index_obs(const std::vector<FeatureObs>& obs) {
  std::unordered_map<TrackId, FeatureObs> m;
  for (const auto& o : obs) m[o.id] = o;
  return m;
}

static std::unordered_map<TrackId, LineObs> index_line_obs(const std::vector<LineObs>& obs) {
  std::unordered_map<TrackId, LineObs> m;
  for (const auto& o : obs) m[o.id] = o;
  return m;
}

// --- Bootstrap Logic ---

static bool estimate_relpose_E(const CameraIntrinsics& K, const Keyframe& kf0,
                               const std::vector<FeatureObs>& obs1,
                               Mat3& R_10, Vec3& t_10, int& inliers_out,
                               int min_inliers, double ransac_thresh_px)
{
  auto o0 = index_obs(kf0.obs);
  std::vector<cv::Point2f> p0_px, p1_px;
  for (const auto& o1 : obs1) {
    auto it = o0.find(o1.id);
    if (it == o0.end()) continue;
    p0_px.emplace_back((float)it->second.px.x(), (float)it->second.px.y());
    p1_px.emplace_back((float)o1.px.x(), (float)o1.px.y());
  }
  if ((int)p0_px.size() < min_inliers) return false;

  cv::Mat Kcv = (cv::Mat_<double>(3,3) << K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1);
  cv::Mat Dcv = cv::Mat::zeros(1,4,CV_64F);
  if (K.has_distortion) for (int i = 0; i < 4; ++i) Dcv.at<double>(0,i) = K.dist[i];

  std::vector<cv::Point2f> p0_n, p1_n;
  cv::undistortPoints(p0_px, p0_n, Kcv, Dcv);
  cv::undistortPoints(p1_px, p1_n, Kcv, Dcv);

  // findEssentialMat for initial motion recovery
  cv::Mat mask;
  cv::Mat E = cv::findEssentialMat(p0_n, p1_n, 1.0, cv::Point2d(0,0), cv::RANSAC, 0.999, ransac_thresh_px / K.fx, mask);
  if (E.empty()) return false;

  cv::Mat R, t;
  inliers_out = cv::recoverPose(E, p0_n, p1_n, R, t, 1.0, cv::Point2d(0,0), mask);
  if (inliers_out < min_inliers) return false;

  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_10(r,c) = R.at<double>(r,c);
  t_10 = Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  if (t_10.norm() > 1e-9) t_10.normalize();
  return true;
}

// --- Frontend Implementation ---

std::vector<Landmark> VoFrontend::collect_local_landmarks() const {
  std::vector<Landmark> lms;
  for (const auto& kv : map_.lms) if (kv.second.valid) lms.push_back(kv.second);
  return lms;
}

bool VoFrontend::should_make_keyframe(const CameraIntrinsics& K,
                                      const std::vector<FeatureObs>& obs,
                                      const std::vector<LineObs>& line_obs,
                                      const Pose& T_wc_est) const {
  if (!last_kf_) return true;
  auto kf_obs = index_obs(last_kf_->obs);
  std::vector<double> pars;
  int total_common = 0;
  
  for (const auto& o : obs) {
    auto it = kf_obs.find(o.id);
    if (it == kf_obs.end()) continue;
    total_common++;
    pars.push_back(parallax_deg(last_kf_->T_wc, T_wc_est, K, it->second.px, o.px));
  }
  
  // Count common lines
  for (const auto& lo : line_obs) {
    for (const auto& kf_lo : last_kf_->line_obs) {
      if (lo.id == kf_lo.id) { total_common++; break; }
    }
  }

  // Median parallax calculation
  double parallax = 0.0;
  if (pars.size() >= 30) {
    std::nth_element(pars.begin(), pars.begin() + pars.size()/2, pars.end());
    parallax = pars[pars.size()/2];
  } else parallax = 1e9;

  return (total_common < p_.min_tracked_features) || 
         (parallax > p_.min_keyframe_parallax_deg) || 
         (frames_since_kf_ > p_.max_frames_between_keyframes);
}

void VoFrontend::create_keyframe(FrameId frame_id, double t_sec, const Pose& T_wc,
                                 const std::vector<FeatureObs>& obs, const std::vector<LineObs>& line_obs) {
  Keyframe kf;
  kf.frame_id = frame_id; kf.t_sec = t_sec; kf.T_wc = T_wc; kf.obs = obs; kf.line_obs = line_obs;
  map_.add_keyframe(kf);
  map_.prune_landmarks();
  last_kf_ = kf;
  frames_since_kf_ = 0;
}

void VoFrontend::triangulate_with_last_keyframe(const CameraIntrinsics& K, const Keyframe& kf0, const Keyframe& kf1) {
  auto o0 = index_obs(kf0.obs);
  auto o1 = index_obs(kf1.obs);
  
  // Point Triangulation
  for (const auto& kv : o0) {
    if (map_.lms.find(kv.first) != map_.lms.end()) continue;
    auto it1 = o1.find(kv.first);
    if (it1 == o1.end()) continue;

    auto tri = triangulate_two_view(kf0.T_wc, kf1.T_wc, K, kv.second.px, it1->second.px, p_.tri);
    if (!tri.ok) continue;

    Landmark lm;
    lm.id = kv.first; lm.p_w = tri.p_w; lm.num_obs = 2; lm.valid = true;
    lm.first_seen_frame = kf1.frame_id; lm.last_seen_frame = kf1.frame_id;
    map_.upsert_landmark(lm);
  }

  // Line Triangulation
  auto lo0 = index_line_obs(kf0.line_obs);
  auto lo1 = index_line_obs(kf1.line_obs);
  for (const auto& kv : lo0) {
    if (map_.lms.find(kv.first) != map_.lms.end()) continue;
    auto it1 = lo1.find(kv.first);
    if (it1 == lo1.end()) continue;

    auto tri = triangulate_line_two_view(kf0.T_wc, kf1.T_wc, K, kv.second, it1->second, p_.tri);
    if (!tri.ok) continue;

    Landmark lm;
    lm.id = kv.first; lm.is_line = true; lm.line = tri.line_w; lm.num_obs = 2; lm.valid = true;
    lm.first_seen_frame = kf1.frame_id; lm.last_seen_frame = kf1.frame_id;
    map_.upsert_landmark(lm);
  }
}

FrontendOutput VoFrontend::process_frame(FrameId frame_id, double t_sec, const CameraIntrinsics& K,
                                         const std::vector<FeatureObs>& obs, const std::vector<LineObs>& line_obs,
                                         const std::optional<Quat>& q_wc_prior) {
  FrontendOutput out;
  out.num_landmarks = (int)map_.lms.size();

  // 1) Initialize first keyframe if starting fresh
  if (!last_pose_) {
    Pose init; if (q_wc_prior) init.q_wc = *q_wc_prior;
    last_pose_ = init;
    create_keyframe(frame_id, t_sec, init, obs, line_obs);
    out.pose_ok = true; out.T_wc = init; out.made_keyframe = true;
    return out;
  }

  frames_since_kf_++;

  // 2) Bootstrap: If map empty, use Essential Matrix to get initial motion
  if (map_.lms.empty() && last_kf_) {
    Mat3 R_10; Vec3 t_10; int inl = 0;
    if (!estimate_relpose_E(K, *last_kf_, obs, R_10, t_10, inl, p_.pnp.min_inliers, p_.pnp.ransac_reproj_thresh_px)) { reset(); out.pose_ok = false; return out; }
    
    Mat3 Rwc1 = last_kf_->T_wc.R_wc() * R_10.transpose();
    Pose T1; T1.q_wc = Quat(Rwc1); T1.p_wc = last_kf_->T_wc.p_wc - (Rwc1 * t_10);
    
    Keyframe prev_kf = *last_kf_;
    create_keyframe(frame_id, t_sec, T1, obs, line_obs);
    triangulate_with_last_keyframe(K, prev_kf, *last_kf_);
    
    if ((int)map_.lms.size() < 30) { reset(); out.pose_ok = false; return out; }
    
    prev_pose_ = last_pose_; last_pose_ = T1;
    out.pose_ok = true; out.T_wc = T1; out.pnp_inliers = inl;
    return out;
  }

  // 3) Normal Tracking: Solve PnP against local map
  int total_common = 0;
  for (const auto& o : obs) if (map_.lms.find(o.id) != map_.lms.end()) total_common++;
  for (const auto& o : line_obs) if (map_.lms.find(o.id) != map_.lms.end()) total_common++;

  if (total_common < p_.pnp.min_inliers) { out.pose_ok = false; return out; }

  Pose init_pose = *last_pose_;
  if (q_wc_prior) init_pose.q_wc = *q_wc_prior;
  auto pnp_res = refine_pose_locally(K, map_, obs, line_obs, p_.pnp, p_.refine, init_pose);

  // Jump check for safety
  if (!pnp_res.ok || (pnp_res.T_wc.p_wc - last_pose_->p_wc).norm() > p_.pnp_max_motion_thresh) { out.pose_ok = false; return out; }

  prev_pose_ = last_pose_;
  last_pose_ = pnp_res.T_wc;
  out.pose_ok = true; out.T_wc = pnp_res.T_wc; out.pnp_inliers = pnp_res.inliers;

  update_landmark_stats_and_prune(map_, frame_id, K, pnp_res.T_wc, obs, p_);

  // 4) Keyframe / Map Management
  if (should_make_keyframe(K, obs, line_obs, pnp_res.T_wc)) {
    Keyframe prev_kf = *last_kf_;
    create_keyframe(frame_id, t_sec, pnp_res.T_wc, obs, line_obs);
    out.made_keyframe = true;
    
    triangulate_new_landmarks:
    triangulate_with_last_keyframe(K, prev_kf, *last_kf_);
    
    if (p_.enable_local_ba) {
      BundleAdjustParams bp;
      bp.window_keyframes = p_.ba_window_keyframes; bp.max_iterations = p_.ba_max_iterations;
      bp.max_cost_increase = p_.ba_max_cost_increase; bp.max_motion_thresh = p_.ba_max_motion_thresh;
      run_local_bundle_adjustment(K, map_, bp);
    }
  }
  out.num_landmarks = (int)map_.lms.size();
  return out;
}

} // namespace stickyvo
