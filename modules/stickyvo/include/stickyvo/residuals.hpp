#pragma once

#include "stickyvo/types.hpp"
#include "stickyvo/camera.hpp"

#if defined(STICKYVO_HAS_CERES) && STICKYVO_HAS_CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace stickyvo {

/**
 * Standard point reprojection residual.
 * Residual = (projected_px - observed_px).
 */
struct ReprojResidual {
  ReprojResidual(double fx, double fy, double cx, double cy, double u, double v)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T* const q_wc_wxyz,
                  const T* const p_wc,
                  const T* const X_w,
                  T* residuals) const {
    // T_cw = T_wc.inverse()
    const T q_cw[4] = {q_wc_wxyz[0], -q_wc_wxyz[1], -q_wc_wxyz[2], -q_wc_wxyz[3]};
    T X_rel[3] = {X_w[0] - p_wc[0], X_w[1] - p_wc[1], X_w[2] - p_wc[2]};
    T X_c[3];
    ceres::QuaternionRotatePoint(q_cw, X_rel, X_c);

    const T z = X_c[2];
    if (z <= T(1e-6)) {
      residuals[0] = T(1000);
      residuals[1] = T(1000);
      return true;
    }

    const T uhat = T(fx_) * (X_c[0] / z) + T(cx_);
    const T vhat = T(fy_) * (X_c[1] / z) + T(cy_);

    residuals[0] = uhat - T(u_);
    residuals[1] = vhat - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(const CameraIntrinsics& K, const Vec2& px) {
    return new ceres::AutoDiffCostFunction<ReprojResidual, 2, 4, 3, 3>(
        new ReprojResidual(K.fx, K.fy, K.cx, K.cy, px.x(), px.y()));
  }

  double fx_, fy_, cx_, cy_;
  double u_, v_;
};

/**
 * Line segment reprojection residual.
 * Minimizes orthogonal distance of endpoints to the observed 2D line.
 */
struct LineReprojResidual {
  LineReprojResidual(double fx, double fy, double cx, double cy, const Vec2& p1, const Vec2& p2)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double len = std::sqrt(dx*dx + dy*dy + 1e-9);
    a_ = -dy / len;
    b_ = dx / len;
    c_ = -(a_ * p1.x() + b_ * p1.y());
  }

  template <typename T>
  bool operator()(const T* const q_wc_wxyz,
                  const T* const p_wc,
                  const T* const L_w_p1,
                  const T* const L_w_p2,
                  T* residuals) const {
    const T q_cw[4] = {q_wc_wxyz[0], -q_wc_wxyz[1], -q_wc_wxyz[2], -q_wc_wxyz[3]};

    auto project = [&](const T* const X_w, T* uv) {
      T X_rel[3] = {X_w[0] - p_wc[0], X_w[1] - p_wc[1], X_w[2] - p_wc[2]};
      T X_c[3];
      ceres::QuaternionRotatePoint(q_cw, X_rel, X_c);
      const T z = X_c[2];
      if (z <= T(1e-6)) {
        uv[0] = T(1e6); uv[1] = T(1e6);
        return false;
      }
      uv[0] = T(fx_) * (X_c[0] / z) + T(cx_);
      uv[1] = T(fy_) * (X_c[1] / z) + T(cy_);
      return true;
    };

    T uv1[2], uv2[2];
    project(L_w_p1, uv1);
    project(L_w_p2, uv2);

    // point-to-line distance residual
    residuals[0] = T(a_) * uv1[0] + T(b_) * uv1[1] + T(c_);
    residuals[1] = T(a_) * uv2[0] + T(b_) * uv2[1] + T(c_);
    return true;
  }

  static ceres::CostFunction* Create(const CameraIntrinsics& K, const LineObs& lo) {
    return new ceres::AutoDiffCostFunction<LineReprojResidual, 2, 4, 3, 3, 3>(
        new LineReprojResidual(K.fx, K.fy, K.cx, K.cy, lo.p1_px, lo.p2_px));
  }

  double fx_, fy_, cx_, cy_;
  double a_, b_, c_;
};

} // namespace stickyvo

#endif
