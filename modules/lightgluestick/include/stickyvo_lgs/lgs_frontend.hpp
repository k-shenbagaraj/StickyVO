#pragma once
#include "stickyvo_lgs/lgs_config.hpp"
#include "stickyvo_lgs/lgs_types.hpp"
#include <memory>

namespace stickyvo_lgs {

/**
 * LgsFrontend invokes a Python-based LightGlue+GlueStick model
 * to obtain high-quality point and line matches between frames.
 */
class LgsFrontend {
public:
  explicit LgsFrontend(const LgsConfig& cfg);
  ~LgsFrontend();

  LgsFrontend(const LgsFrontend&) = delete;
  LgsFrontend& operator=(const LgsFrontend&) = delete;

  /**
   * Run inference on a pair of images.
   */
  PairMatches infer_pair(const ImageView& img0,
                         const ImageView& img1,
                         const CameraIntrinsics& K) const;

  /**
   * Clear any internal temporal state or caches.
   */
  void reset_state();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace stickyvo_lgs
