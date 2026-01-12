#pragma once

#include <Eigen/Core>
#include <vector>
#include <deque>
#include <array>
#include <Eigen/Geometry>
#include <cstdint>
#include <unordered_map>
#include <optional>

namespace stickyvo {

// --- Basic Geometric Types ---
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

using TrackId = std::uint64_t;
using FrameId = std::uint64_t;

struct CameraIntrinsics {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::array<double,4> dist{{0,0,0,0}}; // k1, k2, p1, p2
  bool has_distortion = false;
};

// World-from-camera transformation
struct Pose {
  Vec3 p_wc = Vec3::Zero();
  Quat q_wc = Quat::Identity();

  Mat3 R_wc() const { return q_wc.toRotationMatrix(); }
  Mat3 R_cw() const { return R_wc().transpose(); }
  Vec3 p_cw() const { return -(R_cw() * p_wc); }
};

// --- Feature Observations ---
struct FeatureObs {
  TrackId id = 0;
  Vec2 px = Vec2::Zero();
  Vec2 norm = Vec2::Zero(); // Normalized image coordinates (z=1 plane)
};

enum class FeatureType { POINT = 0, LINE = 1 };

struct Line2D {
  Vec2 p1 = Vec2::Zero();
  Vec2 p2 = Vec2::Zero();
};

struct Line3D {
  Vec3 p1 = Vec3::Zero();
  Vec3 p2 = Vec3::Zero();
};

struct LineObs {
  TrackId id = 0;
  Vec2 p1_px = Vec2::Zero();
  Vec2 p2_px = Vec2::Zero();
  Vec2 p1_norm = Vec2::Zero();
  Vec2 p2_norm = Vec2::Zero();
};

// --- Map Records ---
struct Landmark {
  TrackId id = 0;
  Vec3 p_w = Vec3::Zero();
  int num_obs = 0;
  bool valid = true;

  FrameId first_seen_frame = 0;
  FrameId last_seen_frame = 0;
  int bad_reproj_streak = 0;     // Sequential fails before pruning
  double last_reproj_err_px = 0;
  bool is_line = false;
  Line3D line;
};

// --- VO Outputs ---
struct VoSample {
  double t_sec = 0.0;
  Pose pose;
};

struct VoState {
  Pose pose;
  std::vector<FeatureObs> features;
  std::vector<LineObs> line_features;
};

} // namespace stickyvo
