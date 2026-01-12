#include "stickyvo/triangulation.hpp"
#include "stickyvo/camera.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>

namespace stickyvo {

// --- Utilities ---

static cv::Matx34d P_from_pose(const Pose& T_wc) {
  Mat3 R = T_wc.R_cw(); Vec3 t = -R * T_wc.p_wc;
  cv::Matx34d P;
  for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) P(r,c) = R(r,c); P(r,3) = t(r); }
  return P;
}

double parallax_deg(const Pose& T0, const Pose& T1, const CameraIntrinsics& K, const Vec2& p0, const Vec2& p1) {
  Vec3 r0 = (T0.R_wc() * Vec3((p0.x()-K.cx)/K.fx, (p0.y()-K.cy)/K.fy, 1.0)).normalized();
  Vec3 r1 = (T1.R_wc() * Vec3((p1.x()-K.cx)/K.fx, (p1.y()-K.cy)/K.fy, 1.0)).normalized();
  return std::acos(std::clamp(r0.dot(r1), -1.0, 1.0)) * 180.0 / M_PI;
}

// --- Point Triangulation ---

TriangulationResult triangulate_two_view(const Pose& T0, const Pose& T1, const CameraIntrinsics& K,
                                         const Vec2& px0, const Vec2& px1, const TriangulationParams& tp) {
  TriangulationResult out; out.parallax_deg = parallax_deg(T0, T1, K, px0, px1);
  if (out.parallax_deg < tp.min_parallax_deg) return out;

  // Linear triangulation via SVD (OpenCV implementation)
  std::vector<cv::Point2d> x0{cv::Point2d((px0.x()-K.cx)/K.fx, (px0.y()-K.cy)/K.fy)}, x1{cv::Point2d((px1.x()-K.cx)/K.fx, (px1.y()-K.cy)/K.fy)};
  cv::Mat X4; cv::triangulatePoints(P_from_pose(T0), P_from_pose(T1), x0, x1, X4);
  
  const double w = X4.at<double>(3,0); if (std::abs(w) < 1e-12) return out;
  Vec3 p_w(X4.at<double>(0,0)/w, X4.at<double>(1,0)/w, X4.at<double>(2,0)/w);
  
  // Depth check
  auto depth = [&](const Pose& T){ return (T.R_cw() * (p_w - T.p_wc)).z(); };
  double d0 = depth(T0), d1 = depth(T1);
  if (d0 < tp.min_depth || d1 < tp.min_depth || d0 > tp.max_depth || d1 > tp.max_depth) return out;
  
  // Reprojection check
  auto reproj = [&](const Pose& T, const Vec2& p){
    Vec3 c = T.R_cw() * (p_w - T.p_wc); Vec2 n(c.x()/c.z(), c.y()/c.z());
    return (px_from_norm(n, K) - p).norm();
  };
  out.reproj_err_px = 0.5 * (reproj(T0, px0) + reproj(T1, px1));
  if (out.reproj_err_px > tp.max_reproj_err_px) return out;

  out.ok = true; out.p_w = p_w; return out;
}

// --- Line Triangulation ---

TriangulateLineResult triangulate_line_two_view(const Pose& T0, const Pose& T1, const CameraIntrinsics& K,
                                                const LineObs& lo0, const LineObs& lo1, const TriangulationParams& tp) {
  TriangulateLineResult out;
  
  // 1) Define viewing planes in world frame
  auto get_plane = [&](const Pose& T, const Vec2& p1, const Vec2& p2) {
    Vec3 nc = (Vec3((p1.x()-K.cx)/K.fx, (p1.y()-K.cy)/K.fy, 1.0).cross(Vec3((p2.x()-K.cx)/K.fx, (p2.y()-K.cy)/K.fy, 1.0))).normalized();
    Vec3 nw = T.R_wc() * nc; return std::make_pair(nw, -nw.dot(T.p_wc));
  };
  auto pl0 = get_plane(T0, lo0.p1_px, lo0.p2_px), pl1 = get_plane(T1, lo1.p1_px, lo1.p2_px);
  
  // 2) Intersection line direction
  Vec3 v = pl0.first.cross(pl1.first); double vn = v.norm();
  if (std::asin(std::clamp(vn, 0.0, 1.0)) * 180.0 / M_PI < 0.4) return out;
  v /= vn; 
  
  // 3) Intersection line origin
  Eigen::Matrix3d A; A.row(0) = pl0.first; A.row(1) = pl1.first; A.row(2) = v;
  Vec3 p_on_line = A.colPivHouseholderQr().solve(Vec3(-pl0.second, -pl1.second, 0.0));
  
  // 4) Project rays to 3D line to find endpoints
  auto proj = [&](const Pose& T, const Vec2& px){
    Vec3 C = T.p_wc, W = (T.R_wc() * Vec3((px.x()-K.cx)/K.fx, (px.y()-K.cy)/K.fy, 1.0)).normalized();
    double WV = W.dot(v), denom = 1.0 - WV*WV;
    return (std::abs(denom) < 1e-9) ? p_on_line : Vec3(p_on_line + (WV * W.dot(p_on_line-C) - v.dot(p_on_line-C))/denom * v);
  };
  out.line_w.p1 = proj(T0, lo0.p1_px); out.line_w.p2 = proj(T0, lo0.p2_px);
  
  // 5) Validation
  auto check = [&](const Pose& T, const Vec3& P){ Vec3 c = T.R_cw()*(P-T.p_wc); return c.z() > tp.min_depth && c.z() < tp.max_depth; };
  if (!check(T0, out.line_w.p1) || !check(T0, out.line_w.p2) || !check(T1, out.line_w.p1) || !check(T1, out.line_w.p2)) return out;
  
  auto err = [&](const Pose& T, const Vec3& P, const Vec2& p1, const Vec2& p2){
    Vec3 c = T.R_cw()*(P-T.p_wc); Vec2 n(c.x()/c.z(), c.y()/c.z()); Vec2 p = px_from_norm(n, K), d = p2-p1;
    double l2 = d.squaredNorm(); return (p - (l2 < 1e-9 ? p1 : p1 + std::clamp((p-p1).dot(d)/l2, 0.0, 1.0) * d)).norm();
  };
  out.reproj_err_px = 0.25 * (err(T0, out.line_w.p1, lo0.p1_px, lo0.p2_px)+err(T0, out.line_w.p2, lo0.p1_px, lo0.p2_px)+err(T1, out.line_w.p1, lo1.p1_px, lo1.p2_px)+err(T1, out.line_w.p2, lo1.p1_px, lo1.p2_px));
  if (out.reproj_err_px > tp.max_reproj_err_px) return out;

  out.ok = true; return out;
}

} // namespace stickyvo
