#include "stickyvo/pnp.hpp"
#include "stickyvo/residuals.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>

namespace stickyvo {

static cv::Mat K_cv(const CameraIntrinsics& K) {
  return (cv::Mat_<double>(3,3) << K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1);
}

// --- PnP Entry Point ---

PnPResult estimate_pose_pnp(const CameraIntrinsics& K, const std::vector<Landmark>& lms,
                             const std::vector<FeatureObs>& obs, const std::vector<LineObs>& line_obs,
                             const PnPParams& pp, const std::optional<Pose>& init)
{
  PnPResult out;
  
  // 1) Point correspondences
  std::unordered_map<TrackId, Vec3> lm_by_id;
  for (const auto& lm : lms) if (lm.valid && !lm.is_line) lm_by_id[lm.id] = lm.p_w;

  std::vector<cv::Point3d> Xw;
  std::vector<cv::Point2d> uv;
  for (const auto& o : obs) {
    auto it = lm_by_id.find(o.id);
    if (it != lm_by_id.end()) { 
      Xw.emplace_back(it->second.x(), it->second.y(), it->second.z()); 
      uv.emplace_back(o.px.x(), o.px.y()); 
    }
  }

  // 2) Line correspondences (use endpoints as points for initial RANSAC)
  std::unordered_map<TrackId, Line3D> line_lm_by_id;
  for (const auto& lm : lms) if (lm.valid && lm.is_line) line_lm_by_id[lm.id] = lm.line;
  for (const auto& lo : line_obs) {
    auto it = line_lm_by_id.find(lo.id);
    if (it != line_lm_by_id.end()) {
      Xw.emplace_back(it->second.p1.x(), it->second.p1.y(), it->second.p1.z()); uv.emplace_back(lo.p1_px.x(), lo.p1_px.y());
      Xw.emplace_back(it->second.p2.x(), it->second.p2.y(), it->second.p2.z()); uv.emplace_back(lo.p2_px.x(), lo.p2_px.y());
    }
  }

  if ((int)Xw.size() < pp.min_inliers) return out;

  // 3) RANSAC
  cv::Mat rvec, tvec; bool use_guess = false;
  if (init) {
    Mat3 R_cw = init->R_cw(); Vec3 t_cw = -R_cw * init->p_wc;
    cv::Mat Rcv(3,3,CV_64F); for (int r=0;r<3;r++) for (int c=0;c<3;c++) Rcv.at<double>(r,c) = R_cw(r,c);
    cv::Rodrigues(Rcv, rvec); tvec = (cv::Mat_<double>(3,1) << t_cw.x(), t_cw.y(), t_cw.z());
    use_guess = true;
  }

  cv::Mat distCoeffs = cv::Mat::zeros(1,4,CV_64F);
  if (K.has_distortion) for (int i=0; i<4; ++i) distCoeffs.at<double>(0,i) = K.dist[i];

  cv::Mat inliers_idx;
  bool ok = cv::solvePnPRansac(Xw, uv, K_cv(K), distCoeffs, rvec, tvec, use_guess, pp.ransac_iters, pp.ransac_reproj_thresh_px, pp.ransac_conf, inliers_idx, cv::SOLVEPNP_ITERATIVE);
  
  // Fallback to global PnP if iterative guess failed
  if (use_guess && (!ok || (int)inliers_idx.total() < pp.min_inliers)) {
      cv::Mat r2, t2, i2;
      if (cv::solvePnPRansac(Xw, uv, K_cv(K), distCoeffs, r2, t2, false, pp.ransac_iters, pp.ransac_reproj_thresh_px, pp.ransac_conf, i2, cv::SOLVEPNP_EPNP)) {
          if ((int)i2.total() >= pp.min_inliers) { ok = true; rvec = r2; tvec = t2; inliers_idx = i2; }
      }
  }

  if (!ok || (int)inliers_idx.total() < pp.min_inliers) return out;

  // 4) Convert and Refine
  cv::Mat Rcw; cv::Rodrigues(rvec, Rcw);
  Mat3 R_cw_cv; for (int r=0;r<3;r++) for (int c=0;c<3;c++) R_cw_cv(r,c) = Rcw.at<double>(r,c);
  Mat3 R_wc_cv = R_cw_cv.transpose();
  Pose pnp_pose; pnp_pose.p_wc = -R_wc_cv * Vec3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)); pnp_pose.q_wc = Quat(R_wc_cv);

  out = refine_pose_pnpl(K, lms, obs, line_obs, pnp_pose);
  out.inliers = (int)inliers_idx.total();
  return out;
}

// --- PnPL Refinement ---

PnPResult refine_pose_pnpl(const CameraIntrinsics& K, const std::vector<Landmark>& lms,
                            const std::vector<FeatureObs>& obs, const std::vector<LineObs>& line_obs,
                            const Pose& init_pose)
{
  PnPResult out; out.T_wc = init_pose;
  double q[4] = {init_pose.q_wc.w(), init_pose.q_wc.x(), init_pose.q_wc.y(), init_pose.q_wc.z()};
  double p[3] = {init_pose.p_wc.x(), init_pose.p_wc.y(), init_pose.p_wc.z()};
  ceres::Problem prob;
  
  // Point residuals
  std::unordered_map<TrackId, Vec3> pt_lms; for (const auto& lm : lms) if (lm.valid && !lm.is_line) pt_lms[lm.id] = lm.p_w;
  int n = 0;
  for (const auto& o : obs) {
    auto it = pt_lms.find(o.id); if (it == pt_lms.end()) continue;
    prob.AddResidualBlock(ReprojResidual::Create(K, o.px), new ceres::HuberLoss(2.0), q, p, it->second.data());
    prob.SetParameterBlockConstant(it->second.data()); n++;
  }
  
  // Line residuals
  std::unordered_map<TrackId, Line3D> ln_lms; for (const auto& lm : lms) if (lm.valid && lm.is_line) ln_lms[lm.id] = lm.line;
  for (const auto& lo : line_obs) {
    auto it = ln_lms.find(lo.id); if (it == ln_lms.end()) continue;
    prob.AddResidualBlock(LineReprojResidual::Create(K, lo), new ceres::HuberLoss(2.0), q, p, it->second.p1.data(), it->second.p2.data());
    prob.SetParameterBlockConstant(it->second.p1.data()); prob.SetParameterBlockConstant(it->second.p2.data()); n++;
  }
  
  if (n < 5) { out.ok = true; return out; }
  
  prob.SetManifold(q, new ceres::QuaternionManifold());
  ceres::Solver::Options opt; opt.linear_solver_type = ceres::DENSE_QR; opt.max_num_iterations = 10;
  ceres::Solver::Summary sum; ceres::Solve(opt, &prob, &sum);
  
  out.T_wc.q_wc = Quat(q[0], q[1], q[2], q[3]).normalized();
  out.T_wc.p_wc = Vec3(p[0], p[1], p[2]); out.ok = true;
  return out;
}

} // namespace stickyvo
