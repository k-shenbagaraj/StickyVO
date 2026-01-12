#pragma once
#include <string>

namespace stickyvo_lgs {

/**
 * Configuration for the LightGlueStick frontend.
 */
struct LgsConfig {
  std::string python_module{"lightgluestick_infer"};
  std::string python_func{"infer_pair"};
  
  std::string model_dir{""};
  int max_keypoints{2048};
  bool use_gpu{true};
  bool force_gray{true};
};

} // namespace stickyvo_lgs
