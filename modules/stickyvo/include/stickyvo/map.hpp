#pragma once

#include "types.hpp"
#include "keyframe.hpp"
#include <deque>
#include <unordered_set>

namespace stickyvo {

/**
 * Map holds the window of active keyframes and their 3D landmarks.
 * Provides pruning and lookup utilities.
 */
struct Map {
  std::deque<Keyframe> keyframes;
  std::unordered_map<TrackId, Landmark> lms;
  int max_keyframes = 10;

  void add_keyframe(const Keyframe& kf) {
    keyframes.push_back(kf);
    while ((int)keyframes.size() > max_keyframes) keyframes.pop_front();
  }

  std::optional<Landmark> get_lm(TrackId id) const {
    auto it = lms.find(id);
    if (it == lms.end()) return std::nullopt;
    return it->second;
  }

  Landmark* get_lm_mut(TrackId id) {
    auto it = lms.find(id);
    if (it == lms.end()) return nullptr;
    return &it->second;
  }

  void upsert_landmark(const Landmark& lm) {
    lms[lm.id] = lm;
  }

  // Remove landmarks no longer observed by any keyframe in the window
  void prune_landmarks() {
    std::unordered_set<TrackId> useful;
    for (const auto& kf : keyframes) {
      for (const auto& o : kf.obs) useful.insert(o.id);
      for (const auto& o : kf.line_obs) useful.insert(o.id);
    }
    for (auto it = lms.begin(); it != lms.end(); ) {
      if (useful.find(it->first) == useful.end()) it = lms.erase(it);
      else ++it;
    }
  }
};

} // namespace stickyvo
