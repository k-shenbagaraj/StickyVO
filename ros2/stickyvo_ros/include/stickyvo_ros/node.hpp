#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <opencv2/core.hpp>
#include <deque>
#include <memory>
#include <string>
#include <Eigen/Core>
#include <vector>

#include "stickyvo_lgs/lgs_frontend.hpp"
#include "stickyvo/core.hpp"

namespace stickyvo_ros {

/**
 * StickyVoNode
 * ROS2 wrapper for the StickyVO core. Handles image/IMU subscriptions,
 * undistortion, LGS inference coordination, and VO state management.
 */
class StickyVoNode : public rclcpp::Node {
public:
  StickyVoNode();
  virtual ~StickyVoNode();

private:
  // --- Parameters ---
  std::string topic_image_;
  std::string topic_imu_;

  // --- Intrinsics & Transformation ---
  stickyvo::CameraIntrinsics K_vo_;
  stickyvo_lgs::CameraIntrinsics K_;
  cv::Mat Kcv_, Dcv_; // OpenCV matrices for undistortPoints
  bool have_distortion_ = false;

  // --- AHRS & IMU State ---
  double ahrs_kp_{2.0};
  double ahrs_ki_{0.0};
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};
  double qw_{1.0}, qx_{0.0}, qy_{0.0}, qz_{0.0}; // Integrated orientation
  double ex_int_{0.0}, ey_int_{0.0}, ez_int_{0.0}; // Error integrators

  // --- Image Processing State ---
  rclcpp::Time last_img_stamp_{0, 0, RCL_ROS_TIME};
  cv::Mat last_gray_;
  int bootstrap_fail_count_{0};
  int tracking_fail_count_{0};

  // --- Components ---
  stickyvo::StickyVoCore core_;
  std::unique_ptr<stickyvo_lgs::LgsFrontend> lgs_;

  // --- Subscriptions ---
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

  // --- Callbacks ---
  void on_image_raw(const sensor_msgs::msg::Image::SharedPtr msg);
  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg);

  // --- Internal Helpers ---
  static cv::Mat to_gray8(const sensor_msgs::msg::Image& msg);
  static stickyvo_lgs::ImageView mat_to_imageview_gray8(const cv::Mat& gray);
  
  void ahrs_update(double dt,
                   double gx, double gy, double gz,
                   double ax, double ay, double az);
};

} // namespace stickyvo_ros
