#include "stickyvo/track_db.hpp"
#include <unordered_set>

namespace stickyvo {

TrackDB::FrameTracks TrackDB::ingest_frame(FrameId frame_id, double t_sec, const CameraIntrinsics& K,
                                           const std::vector<Vec2>& kpts_px, const std::vector<Eigen::Vector2i>& matches) {
  FrameTracks out; out.frame_id = frame_id; out.t_sec = t_sec; out.obs.reserve(kpts_px.size());
  std::unordered_map<int, TrackId> curr; std::unordered_set<int> used_prev;
  
  // 1) Propagate IDs from previous frame
  for (const auto& m : matches) {
    if (m.y() < 0 || m.y() >= (int)kpts_px.size() || m.x() < 0 || !used_prev.insert(m.x()).second) continue;
    auto it = last_kpidx_to_id_.find(m.x());
    if (it != last_kpidx_to_id_.end()) curr.emplace(m.y(), it->second);
  }
  
  // 2) Assign new IDs to fresh keypoints
  for (int i = 0; i < (int)kpts_px.size(); ++i) {
    auto it = curr.find(i); TrackId id = (it == curr.end()) ? (curr[i] = next_id_++) : it->second;
    FeatureObs fo; fo.id = id; fo.px = kpts_px[i]; fo.norm = norm_from_px(fo.px, K);
    out.obs.push_back(fo);
  }
  
  last_kpidx_to_id_ = curr; last_ = out; return out;
}

std::vector<LineObs> TrackDB::ingest_lines(FrameId frame_id, double t_sec, const CameraIntrinsics& K,
                                           const std::vector<Line2D>& lines_px, const std::vector<Eigen::Vector2i>& matches) {
  std::vector<LineObs> out; out.reserve(lines_px.size());
  std::unordered_map<int, TrackId> curr; std::unordered_set<int> used_prev;
  
  // Propagate line IDs
  for (const auto& m : matches) {
    if (m.y() < 0 || m.y() >= (int)lines_px.size() || m.x() < 0 || !used_prev.insert(m.x()).second) continue;
    auto it = last_lineidx_to_id_.find(m.x());
    if (it != last_lineidx_to_id_.end()) curr.emplace(m.y(), it->second);
  }
  
  // Assign new IDs
  for (int i = 0; i < (int)lines_px.size(); ++i) {
    auto it = curr.find(i); TrackId id = (it == curr.end()) ? (curr[i] = next_id_++) : it->second;
    LineObs lo; lo.id = id; lo.p1_px = lines_px[i].p1; lo.p2_px = lines_px[i].p2; lo.p1_norm = norm_from_px(lo.p1_px, K); lo.p2_norm = norm_from_px(lo.p2_px, K);
    out.push_back(lo);
  }
  
  last_lineidx_to_id_ = curr; return out;
}

} // namespace stickyvo
