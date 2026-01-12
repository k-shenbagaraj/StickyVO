#pragma once
#include "types.hpp"

namespace stickyvo {

inline Vec2 norm_from_px(const Vec2& u_px, const CameraIntrinsics& K) {
  return Vec2((u_px.x() - K.cx) / K.fx, (u_px.y() - K.cy) / K.fy);
}

inline Vec2 px_from_norm(const Vec2& u_norm, const CameraIntrinsics& K) {
  return Vec2(u_norm.x() * K.fx + K.cx, u_norm.y() * K.fy + K.cy);
}

} // namespace stickyvo
