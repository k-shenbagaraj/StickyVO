#include "stickyvo/core.hpp"
#include "stickyvo/track_db.hpp"
#include "stickyvo/vo_frontend.hpp"
#include <cstdio>
#include <cmath>

namespace stickyvo {

// --- Parameter Mapping ---

static stickyvo::FrontendParams make_frontend_params_from(const StickyVoCore::Params& p) {
  stickyvo::FrontendParams fp;
  
  // Tracking thresholds
  fp.min_tracked_features = p.min_tracked_features;
  fp.min_keyframe_parallax_deg = p.min_keyframe_parallax_deg;
  fp.max_frames_between_keyframes = p.max_frames_between_keyframes;
  
  // Triangulation thresholds
  fp.tri.min_parallax_deg = p.tri_min_parallax_deg;
  fp.tri.max_reproj_err_px = p.tri_max_reproj_err_px;
  fp.tri.min_depth = 0.1;
  fp.tri.max_depth = 1000.0;
  
  // PnP thresholds
  fp.pnp.min_inliers = p.min_inliers;
  fp.pnp.ransac_reproj_thresh_px = p.ransac_thresh_px;
  
  // Optimizer thresholds
  fp.map_max_keyframes = p.map_max_keyframes;
  fp.ba_max_cost_increase = p.ba_max_cost_increase;
  fp.ba_max_motion_thresh = p.ba_max_motion_thresh;
  fp.pnp_max_motion_thresh = p.pnp_max_motion_thresh;
  
  return fp;
}

// --- Lifecycle ---

StickyVoCore::StickyVoCore(const Params& p)
: p_(p), state_(), frontend_(make_frontend_params_from(p)) {
  reset();
}

void StickyVoCore::reset() {
  state_ = VoState{};
  state_.pose = Pose{};
  state_.features.clear();
  vo_hist_.clear();
  trackdb_ = stickyvo::TrackDB{}; 
  frontend_.reset();
  bootstrapped_ = false;
  frame_id_ = 0;
}

// --- Math Utilities ---

Eigen::Vector2d StickyVoCore::norm_from_px(const Eigen::Vector2d& u, const CameraIntrinsics& K) {
  return Eigen::Vector2d((u.x() - K.cx) / K.fx, (u.y() - K.cy) / K.fy);
}

std::optional<size_t> StickyVoCore::nearest_vo_index(double t_sec, const std::deque<VoSample>& hist) {
  if (hist.empty()) return std::nullopt;
  size_t lo = 0, hi = hist.size() - 1;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (hist[mid].t_sec < t_sec) lo = mid + 1;
    else hi = mid;
  }
  size_t best = lo;
  if (best > 0) {
    const double d0 = std::abs(hist[best].t_sec - t_sec);
    const double d1 = std::abs(hist[best - 1].t_sec - t_sec);
    if (d1 < d0) best = best - 1;
  }
  return best;
}

// --- Main Pipeline ---

bool StickyVoCore::process_and_update(double t0, double t1,
                                      const CameraIntrinsics& K,
                                      const PairMatchesLite& pm,
                                      const std::optional<Quat>& q_wc_prior)
{
  stickyvo::CameraIntrinsics K2;
  K2.fx = K.fx; K2.fy = K.fy; K2.cx = K.cx; K2.cy = K.cy;

  // Transaction snapshot for rollback on failure
  const auto trackdb_backup = trackdb_;
  const auto state_backup   = state_;
  const auto frame_id_backup = frame_id_;
  const auto bootstrapped_backup = bootstrapped_;

  // 1) Internal Bootstrap: If not yet tracked, use first frame as origin
  if (!bootstrapped_) {
    std::vector<Eigen::Vector2i> empty_matches;
    std::vector<stickyvo::Vec2> kpts0;
    kpts0.reserve(pm.kpts0_px.size());
    for (const auto& u : pm.kpts0_px) kpts0.emplace_back(u.x(), u.y());

    auto ft0 = trackdb_.ingest_frame(frame_id_++, t0, K2, kpts0, empty_matches);
    
    std::vector<stickyvo::Line2D> lines0;
    lines0.reserve(pm.lines0_px.size());
    for (const auto& l : pm.lines0_px) lines0.push_back(l);
    std::vector<Eigen::Vector2i> empty_line_matches;
    auto lobs0 = trackdb_.ingest_lines(ft0.frame_id, ft0.t_sec, K2, lines0, empty_line_matches);

    // Initial keyframe creation
    auto out0 = frontend_.process_frame(ft0.frame_id, ft0.t_sec, K2, ft0.obs, lobs0, q_wc_prior);
    if (out0.made_keyframe) bootstrapped_ = true;
  }

  // 2) Tracking: Ingest current frame matches
  std::vector<stickyvo::Vec2> kpts1;
  kpts1.reserve(pm.kpts1_px.size());
  for (const auto& u : pm.kpts1_px) kpts1.emplace_back(u.x(), u.y());

  std::vector<Eigen::Vector2i> matches_prev_to_curr;
  matches_prev_to_curr.reserve(pm.matches.size());
  for (const auto& m : pm.matches) matches_prev_to_curr.emplace_back(m.x(), m.y());

  auto ft1 = trackdb_.ingest_frame(frame_id_++, t1, K2, kpts1, matches_prev_to_curr);

  std::vector<stickyvo::Line2D> lines1;
  lines1.reserve(pm.lines1_px.size());
  for (const auto& l : pm.lines1_px) lines1.push_back(l);
  
  std::vector<Eigen::Vector2i> line_matches;
  line_matches.reserve(pm.line_matches.size());
  for (const auto& m : pm.line_matches) line_matches.emplace_back(m.x(), m.y());

  auto lobs1 = trackdb_.ingest_lines(ft1.frame_id, ft1.t_sec, K2, lines1, line_matches);

  state_.features = ft1.obs;
  state_.line_features = lobs1;

  // 3) Frontend Processing: PnP, Keyframing, Triangulation, BA
  auto out1 = frontend_.process_frame(ft1.frame_id, ft1.t_sec, K2, ft1.obs, lobs1, q_wc_prior);

  if (!out1.pose_ok) {
    // Rollback state if tracking failed
    trackdb_     = trackdb_backup;
    state_       = state_backup;
    frame_id_    = frame_id_backup;
    bootstrapped_ = bootstrapped_backup;
    return false;
  }

  // Commit update
  state_.pose.p_wc = out1.T_wc.p_wc;
  state_.pose.q_wc = out1.T_wc.q_wc;

  // Update history
  vo_hist_.push_back(VoSample{t1, state_.pose});
  while ((int)vo_hist_.size() > p_.pose_history_max) vo_hist_.pop_front();

  return true;
}

} // namespace stickyvo
