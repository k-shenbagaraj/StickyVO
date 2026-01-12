#pragma once
#include <cmath>

namespace stickyvo_ros {

/**
 * Minimal quaternion structure with normalization.
 */
struct Quat {
  double w{1.0}, x{0.0}, y{0.0}, z{0.0};

  void normalize() {
    const double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n > 0.0) { w/=n; x/=n; y/=n; z/=n; }
  }
};

/**
 * Mahony AHRS Filter
 * Stabilizes roll and pitch using accelerometer feedback.
 * Yaw is unobserved and will drift.
 */
class MahonyAhrs {
public:
  MahonyAhrs(double kp, double ki) : kp_(kp), ki_(ki) {}

  /**
   * Update internal orientation estimate.
   * @param gx, gy, gz Gyroscope readings (rad/s)
   * @param ax, ay, az Accelerometer readings (m/s^2)
   * @param dt_s Time step (seconds)
   */
  void update(double gx, double gy, double gz,
              double ax, double ay, double az,
              double dt_s)
  {
    if (dt_s <= 0.0) return;

    const double an = std::sqrt(ax*ax + ay*ay + az*az);
    if (an < 1e-6) {
      integrate_gyro(gx, gy, gz, dt_s);
      return;
    }
    ax /= an; ay /= an; az /= an;

    // Body gravity direction from current orientation
    const double vx = 2.0*(q_.x*q_.z - q_.w*q_.y);
    const double vy = 2.0*(q_.w*q_.x + q_.y*q_.z);
    const double vz = q_.w*q_.w - q_.x*q_.x - q_.y*q_.y + q_.z*q_.z;

    // Gravity error (measured vs estimated)
    const double ex = (ay*vz - az*vy);
    const double ey = (az*vx - ax*vz);
    const double ez = (ax*vy - ay*vx);

    if (ki_ > 0.0) {
      ix_ += ki_ * ex * dt_s;
      iy_ += ki_ * ey * dt_s;
      iz_ += ki_ * ez * dt_s;
    }

    gx += kp_ * ex + ix_;
    gy += kp_ * ey + iy_;
    gz += kp_ * ez + iz_;

    integrate_gyro(gx, gy, gz, dt_s);
  }

  const Quat& quat() const { return q_; }

private:
  void integrate_gyro(double gx, double gy, double gz, double dt_s) {
    const double h = 0.5 * dt_s;
    const double qw = q_.w, qx = q_.x, qy = q_.y, qz = q_.z;

    q_.w += (-qx*gx - qy*gy - qz*gz) * h;
    q_.x += ( qw*gx + qy*gz - qz*gy) * h;
    q_.y += ( qw*gy - qx*gz + qz*gx) * h;
    q_.z += ( qw*gz + qx*gy - qy*gx) * h;

    q_.normalize();
  }

  double kp_{2.0};
  double ki_{0.0};
  double ix_{0.0}, iy_{0.0}, iz_{0.0};
  Quat q_{};
};

} // namespace stickyvo_ros
