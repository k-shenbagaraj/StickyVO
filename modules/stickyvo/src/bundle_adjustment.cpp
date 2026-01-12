#include "stickyvo/bundle_adjustment.hpp"
#include "stickyvo/residuals.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>

#if defined(STICKYVO_HAS_CERES) && STICKYVO_HAS_CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>

namespace stickyvo {

bool run_local_bundle_adjustment(const CameraIntrinsics& K, Map& map, const BundleAdjustParams& bp) {
  if ((int)map.keyframes.size() < 2) return false;
  
  // Define sliding window
  const int window = std::max(2, std::min(bp.window_keyframes, (int)map.keyframes.size()));
  const int start_idx = (int)map.keyframes.size() - window;

  // 1) Collect landmarks observed in the window
  std::unordered_set<TrackId> lm_ids, line_lm_ids;
  for (int i = start_idx; i < (int)map.keyframes.size(); ++i) {
    const auto& kf = map.keyframes[(size_t)i];
    for (const auto& o : kf.obs) {
      auto it = map.lms.find(o.id);
      if (it != map.lms.end() && it->second.valid && !it->second.is_line) lm_ids.insert(o.id);
    }
    for (const auto& o : kf.line_obs) {
      auto it = map.lms.find(o.id);
      if (it != map.lms.end() && it->second.valid && it->second.is_line) line_lm_ids.insert(o.id);
    }
  }
  if (lm_ids.size() + line_lm_ids.size() < 20) return false;

  ceres::Problem problem;
  
  // 2) Add pose parameter blocks
  struct PoseBlock { double q[4], p[3]; };
  std::vector<PoseBlock> poses((size_t)window);

  for (int wi = 0; wi < window; ++wi) {
    const auto& kf = map.keyframes[(size_t)(start_idx + wi)];
    poses[(size_t)wi].q[0] = kf.T_wc.q_wc.w(); poses[(size_t)wi].q[1] = kf.T_wc.q_wc.x();
    poses[(size_t)wi].q[2] = kf.T_wc.q_wc.y(); poses[(size_t)wi].q[3] = kf.T_wc.q_wc.z();
    poses[(size_t)wi].p[0] = kf.T_wc.p_wc.x(); poses[(size_t)wi].p[1] = kf.T_wc.p_wc.y(); poses[(size_t)wi].p[2] = kf.T_wc.p_wc.z();
    
    problem.AddParameterBlock(poses[(size_t)wi].q, 4);
    problem.SetManifold(poses[(size_t)wi].q, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(poses[(size_t)wi].p, 3);
  }
  
  // Fix oldest pose to remove gauge freedom
  problem.SetParameterBlockConstant(poses[0].q);
  problem.SetParameterBlockConstant(poses[0].p);

  // 3) Add landmark parameter blocks
  std::unordered_map<TrackId, std::array<double, 3>> lm_params;
  for (TrackId id : lm_ids) {
    auto it = map.lms.find(id); if (it == map.lms.end()) continue;
    auto& arr = lm_params[id];
    arr[0] = it->second.p_w.x(); arr[1] = it->second.p_w.y(); arr[2] = it->second.p_w.z();
    problem.AddParameterBlock(arr.data(), 3);
  }
  
  std::unordered_map<TrackId, std::array<double, 6>> line_lm_params;
  for (TrackId id : line_lm_ids) {
    auto it = map.lms.find(id); if (it == map.lms.end()) continue;
    auto& arr = line_lm_params[id];
    arr[0] = it->second.line.p1.x(); arr[1] = it->second.line.p1.y(); arr[2] = it->second.line.p1.z();
    arr[3] = it->second.line.p2.x(); arr[4] = it->second.line.p2.y(); arr[5] = it->second.line.p2.z();
    problem.AddParameterBlock(arr.data(), 3); problem.AddParameterBlock(arr.data() + 3, 3);
  }

  // 4) Add reprojection residuals
  int residual_count = 0;
  for (int wi = 0; wi < window; ++wi) {
    const auto& kf = map.keyframes[(size_t)(start_idx + wi)];
    // Point residuals
    for (const auto& o : kf.obs) {
      auto pit = lm_params.find(o.id); if (pit == lm_params.end()) continue;
      problem.AddResidualBlock(ReprojResidual::Create(K, o.px), 
                               new ceres::HuberLoss(bp.huber_delta_px), 
                               poses[(size_t)wi].q, poses[(size_t)wi].p, pit->second.data());
      residual_count++;
    }
    // Line residuals
    for (const auto& o : kf.line_obs) {
      auto pit = line_lm_params.find(o.id); if (pit == line_lm_params.end()) continue;
      problem.AddResidualBlock(LineReprojResidual::Create(K, o), 
                               new ceres::HuberLoss(bp.huber_delta_px), 
                               poses[(size_t)wi].q, poses[(size_t)wi].p, pit->second.data(), pit->second.data() + 3);
      residual_count++;
    }
  }
  if (residual_count < 50) return false;

  // 5) Solve
  ceres::Solver::Options options;
  options.max_num_iterations = bp.max_iterations; 
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 4;
  options.function_tolerance = options.gradient_tolerance = options.parameter_tolerance = 1e-4;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  // 6) Post-BA Safety Checks
  if (summary.final_cost > summary.initial_cost * bp.max_cost_increase) return false;

  for (int wi = 0; wi < window; ++wi) {
    const auto& kf_orig = map.keyframes[(size_t)(start_idx + wi)];
    if ((Vec3(poses[(size_t)wi].p[0], poses[(size_t)wi].p[1], poses[(size_t)wi].p[2]) - kf_orig.T_wc.p_wc).norm() > bp.max_motion_thresh) return false;
  }

  // 7) Write back results
  for (int wi = 0; wi < window; ++wi) {
    auto& kf = map.keyframes[(size_t)(start_idx + wi)];
    kf.T_wc.q_wc = Quat(poses[(size_t)wi].q[0], poses[(size_t)wi].q[1], poses[(size_t)wi].q[2], poses[(size_t)wi].q[3]);
    kf.T_wc.q_wc.normalize(); kf.T_wc.p_wc = Vec3(poses[(size_t)wi].p[0], poses[(size_t)wi].p[1], poses[(size_t)wi].p[2]);
  }
  for (auto& kv : lm_params) { auto it = map.lms.find(kv.first); if (it != map.lms.end()) it->second.p_w = Vec3(kv.second[0], kv.second[1], kv.second[2]); }
  for (auto& kv : line_lm_params) {
    auto it = map.lms.find(kv.first); if (it != map.lms.end()) {
      it->second.line.p1 = Vec3(kv.second[0], kv.second[1], kv.second[2]);
      it->second.line.p2 = Vec3(kv.second[3], kv.second[4], kv.second[5]);
    }
  }
  return true;
}

} // namespace stickyvo

#else

namespace stickyvo {
bool run_local_bundle_adjustment(const CameraIntrinsics&, Map&, const BundleAdjustParams&) { return false; }
} // namespace stickyvo

#endif
