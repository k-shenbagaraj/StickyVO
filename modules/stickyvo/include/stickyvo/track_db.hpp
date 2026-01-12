#pragma once

#include "types.hpp"
#include "camera.hpp"
#include <Eigen/Core>
#include <vector>

namespace stickyvo {

/**
 * TrackDB assigns and persists stable TrackIds across frames.
 * Links raw correspondences from the frontend into long-term features.
 */
class TrackDB {
public:
  TrackDB() = default;

  struct FrameTracks {
    FrameId frame_id = 0;
    double t_sec = 0.0;
    std::vector<FeatureObs> obs;
  };

  // Convert raw matches into FeatureObs with persistent TrackIds
  FrameTracks ingest_frame(
      FrameId frame_id, double t_sec,
      const CameraIntrinsics& K,
      const std::vector<Vec2>& kpts_px,
      const std::vector<Eigen::Vector2i>& matches_prev_to_curr);

  const FrameTracks* last_frame() const { return last_ ? &(*last_) : nullptr; }
  
  // Similarly ingest line correspondences
  std::vector<LineObs> ingest_lines(
      FrameId frame_id, double t_sec,
      const CameraIntrinsics& K,
      const std::vector<Line2D>& lines_px,
      const std::vector<Eigen::Vector2i>& matches_prev_to_curr);

private:
  TrackId next_id_ = 1;

  // Mapping from previous frame keypoint index to persistent TrackId
  std::unordered_map<int, TrackId> last_kpidx_to_id_;
  std::optional<FrameTracks> last_;
  std::unordered_map<int, TrackId> last_lineidx_to_id_;
};

} // namespace stickyvo
