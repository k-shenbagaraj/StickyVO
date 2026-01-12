#include "stickyvo_ros/node.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace stickyvo_ros {

// --- Geometric Utilities ---

static void undistort_px_points(const cv::Mat& Kcv,
                                const cv::Mat& Dcv,
                                const std::vector<Eigen::Vector2d>& in_px,
                                std::vector<Eigen::Vector2d>& out_px)
{
  out_px.clear();
  out_px.reserve(in_px.size());
  if (in_px.empty()) return;

  std::vector<cv::Point2f> src;
  src.reserve(in_px.size());
  for (const auto& p : in_px) src.emplace_back((float)p.x(), (float)p.y());

  std::vector<cv::Point2f> dst;
  cv::undistortPoints(src, dst, Kcv, Dcv, cv::noArray(), Kcv);

  for (const auto& p : dst) out_px.emplace_back((double)p.x, (double)p.y);
}

// --- Lifecycle & Initialization ---

StickyVoNode::StickyVoNode()
: rclcpp::Node("stickyvo_node")
{
  // 1) Parameter Declaration
  this->declare_parameter<std::string>("topic_image_compressed", "/cam_driver/image_raw");
  this->declare_parameter<std::string>("topic_imu", "/vectornav/imu");
  this->declare_parameter<bool>("cam0.is_compressed", false);
  this->declare_parameter<std::vector<double>>("cam0.intrinsics", std::vector<double>{});
  this->declare_parameter<std::vector<int64_t>>("cam0.resolution", std::vector<int64_t>{});
  this->declare_parameter<std::vector<double>>("cam0.distortion_coeffs", std::vector<double>{});
  this->declare_parameter<double>("ahrs.kp", 2.0);
  this->declare_parameter<double>("ahrs.ki", 0.0);
  this->declare_parameter<int64_t>("imu_buffer_max", 4000);

  // VO Parameters
  this->declare_parameter<int64_t>("vo.min_inliers", 20);
  this->declare_parameter<double>("vo.ransac_thresh_px", 5.0);
  this->declare_parameter<int64_t>("vo.min_tracked_features", 150);
  this->declare_parameter<double>("vo.min_keyframe_parallax_deg", 2.0);
  this->declare_parameter<int64_t>("vo.max_frames_between_keyframes", 30);
  this->declare_parameter<int64_t>("vo.map_max_keyframes", 10);
  this->declare_parameter<double>("vo.tri_min_parallax_deg", 1.0);
  this->declare_parameter<double>("vo.ba_max_motion_thresh", 5.0);

  // 2) Parameter Loading
  topic_image_ = this->get_parameter("topic_image_compressed").as_string();
  topic_imu_   = this->get_parameter("topic_imu").as_string();
  ahrs_kp_ = this->get_parameter("ahrs.kp").as_double();
  ahrs_ki_ = this->get_parameter("ahrs.ki").as_double();

  auto intr = this->get_parameter("cam0.intrinsics").as_double_array();
  if (intr.size() == 4) {
    K_.fx = intr[0]; K_.fy = intr[1]; K_.cx = intr[2]; K_.cy = intr[3];
    K_vo_.fx = intr[0]; K_vo_.fy = intr[1]; K_vo_.cx = intr[2]; K_vo_.cy = intr[3];
    K_vo_.has_distortion = false;
  }
  
  auto dist = this->get_parameter("cam0.distortion_coeffs").as_double_array();
  Kcv_ = (cv::Mat_<double>(3,3) << K_vo_.fx, 0, K_vo_.cx, 0, K_vo_.fy, K_vo_.cy, 0, 0, 1);
  if (dist.size() >= 4) {
    Dcv_ = cv::Mat::zeros(1, 5, CV_64F);
    for (size_t i=0; i<std::min(dist.size(), (size_t)5); ++i) Dcv_.at<double>(0,i) = dist[i];
    have_distortion_ = true;
  }

  // 3) Component Initialization
  try {
    stickyvo_lgs::LgsConfig cfg;
    cfg.python_module = "lgs_py_bridge";
    cfg.python_func   = "infer_pair";
    cfg.use_gpu       = true;
    lgs_ = std::make_unique<stickyvo_lgs::LgsFrontend>(cfg);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "LGS construction failed: %s", e.what());
  }

  stickyvo::StickyVoCore::Params cop;
  cop.min_inliers = static_cast<int>(this->get_parameter("vo.min_inliers").as_int());
  cop.ransac_thresh_px = this->get_parameter("vo.ransac_thresh_px").as_double();
  cop.min_tracked_features = static_cast<int>(this->get_parameter("vo.min_tracked_features").as_int());
  cop.min_keyframe_parallax_deg = this->get_parameter("vo.min_keyframe_parallax_deg").as_double();
  cop.max_frames_between_keyframes = static_cast<int>(this->get_parameter("vo.max_frames_between_keyframes").as_int());
  cop.map_max_keyframes = static_cast<int>(this->get_parameter("vo.map_max_keyframes").as_int());
  cop.tri_min_parallax_deg = this->get_parameter("vo.tri_min_parallax_deg").as_double();
  cop.ba_max_motion_thresh = this->get_parameter("vo.ba_max_motion_thresh").as_double();
  core_ = stickyvo::StickyVoCore(cop);

  // 4) Pub/Sub
  sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(topic_image_, rclcpp::SensorDataQoS(), std::bind(&StickyVoNode::on_image_raw, this, std::placeholders::_1));
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(topic_imu_, rclcpp::SensorDataQoS(), std::bind(&StickyVoNode::on_imu, this, std::placeholders::_1));
}

StickyVoNode::~StickyVoNode() {
  sub_img_.reset();
  sub_imu_.reset();
  lgs_.reset();
}

// --- Image Processing Pipeline ---

cv::Mat StickyVoNode::to_gray8(const sensor_msgs::msg::Image& msg) {
  if (msg.data.empty()) throw std::runtime_error("Empty image");
  
  if (msg.encoding == "mono8") return cv::Mat((int)msg.height, (int)msg.width, CV_8UC1, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step).clone();
  
  cv::Mat bgr;
  if (msg.encoding == "bgr8") bgr = cv::Mat((int)msg.height, (int)msg.width, CV_8UC3, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
  else if (msg.encoding == "rgb8") {
    cv::Mat rgb = cv::Mat((int)msg.height, (int)msg.width, CV_8UC3, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  }
  else if (msg.encoding.find("bayer") != std::string::npos) {
    cv::Mat bayer((int)msg.height, (int)msg.width, CV_8UC1, const_cast<uint8_t*>(msg.data.data()), (size_t)msg.step);
    if (msg.encoding == "bayer_rggb8")      cv::cvtColor(bayer, bgr, cv::COLOR_BayerRG2BGR);
    else if (msg.encoding == "bayer_bggr8") cv::cvtColor(bayer, bgr, cv::COLOR_BayerBG2BGR);
    else if (msg.encoding == "bayer_gbrg8") cv::cvtColor(bayer, bgr, cv::COLOR_BayerGB2BGR);
    else                                    cv::cvtColor(bayer, bgr, cv::COLOR_BayerGR2BGR);
  } else throw std::runtime_error("Unsupported encoding: " + msg.encoding);

  cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

void StickyVoNode::on_image_raw(const sensor_msgs::msg::Image::SharedPtr msg) {
  cv::Mat gray; try { gray = to_gray8(*msg); } catch (...) { return; }

  const rclcpp::Time t = rclcpp::Time(msg->header.stamp);
  if (last_gray_.empty()) {
    last_gray_ = gray; last_img_stamp_ = t;
    return;
  }

  if (!lgs_) { last_gray_ = gray; last_img_stamp_ = t; return; }

  try {
    stickyvo_lgs::ImageView v0{(uint8_t*)last_gray_.data, last_gray_.cols, last_gray_.rows, (int)last_gray_.step, stickyvo_lgs::ImageView::Format::kGray8};
    stickyvo_lgs::ImageView v1{(uint8_t*)gray.data, gray.cols, gray.rows, (int)gray.step, stickyvo_lgs::ImageView::Format::kGray8};
    auto matches = lgs_->infer_pair(v0, v1, K_);

    // Match assembly & Undistort
    stickyvo::PairMatchesLite pm;
    pm.score = matches.score;
    std::vector<Eigen::Vector2d> p0_px, p1_px;
    for (const auto& k : matches.f0.keypoints) p0_px.emplace_back(k.x, k.y);
    for (const auto& k : matches.f1.keypoints) p1_px.emplace_back(k.x, k.y);

    if (have_distortion_) {
      undistort_px_points(Kcv_, Dcv_, p0_px, pm.kpts0_px);
      undistort_px_points(Kcv_, Dcv_, p1_px, pm.kpts1_px);
    } else { pm.kpts0_px = p0_px; pm.kpts1_px = p1_px; }

    for (const auto& m : matches.point_matches) pm.matches.emplace_back(m.i0, m.i1);
    
    // Line assembly & Undistort
    auto to_sl = [](const stickyvo_lgs::Line2D& l){ return stickyvo::Line2D{{l.p0.x, l.p0.y}, {l.p1.x, l.p1.y}}; };
    std::vector<stickyvo::Line2D> l0_px, l1_px;
    for (const auto& l : matches.f0.lines) l0_px.push_back(to_sl(l));
    for (const auto& l : matches.f1.lines) l1_px.push_back(to_sl(l));

    auto und_lines = [&](const std::vector<stickyvo::Line2D>& in, std::vector<stickyvo::Line2D>& out) {
      std::vector<Eigen::Vector2d> pts, pts_ud;
      for (const auto& l : in) { pts.push_back(l.p1); pts.push_back(l.p2); }
      if (have_distortion_) undistort_px_points(Kcv_, Dcv_, pts, pts_ud); else pts_ud = pts;
      for (size_t i=0; i<in.size(); ++i) out.push_back({pts_ud[2*i], pts_ud[2*i+1]});
    };
    und_lines(l0_px, pm.lines0_px); und_lines(l1_px, pm.lines1_px);
    for (const auto& m : matches.line_matches) pm.line_matches.emplace_back(m.i0, m.i1);

    // VO Update
    stickyvo::CameraIntrinsics K_vo = K_vo_; K_vo.has_distortion = false;
    const bool ok = core_.process_and_update(last_img_stamp_.seconds(), t.seconds(), K_vo, pm, stickyvo::Quat(qw_, qx_, qy_, qz_));

    // Recovery logic
    if (!ok) {
      if (!core_.is_bootstrapped()) {
        bootstrap_fail_count_++;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "VO: Bootstrap failed (%d/30)", bootstrap_fail_count_);
        if (bootstrap_fail_count_ > 30) { 
          RCLCPP_ERROR(get_logger(), "VO: Bootstrap timed out. Resetting.");
          core_.reset(); bootstrap_fail_count_ = 0; 
        }
        else lgs_->reset_state();
      } else {
        tracking_fail_count_++;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "VO: Tracking failed (%d/10)", tracking_fail_count_);
        if (tracking_fail_count_ > 10) { 
          RCLCPP_ERROR(get_logger(), "VO: Tracking LOST. Resetting.");
          core_.reset(); tracking_fail_count_ = 0; 
        }
      }
    } else {
      bootstrap_fail_count_ = tracking_fail_count_ = 0;
      const auto& st = core_.state();
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100, "VO: p=(%.2f %.2f %.2f) q=(%.2f %.2f %.2f %.2f) tracks=%zu", 
                           st.pose.p_wc.x(), st.pose.p_wc.y(), st.pose.p_wc.z(), st.pose.q_wc.w(), st.pose.q_wc.x(), st.pose.q_wc.y(), st.pose.q_wc.z(), st.features.size());
    }
  } catch (...) {}

  last_gray_ = gray; last_img_stamp_ = t;
}

// --- IMU & Orientation Monitoring ---

void StickyVoNode::on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
  const rclcpp::Time t = rclcpp::Time(msg->header.stamp);
  double dt = (last_imu_stamp_.nanoseconds() != 0) ? (t - last_imu_stamp_).seconds() : 0.0;
  if (dt < 0.0 || dt > 0.1) dt = 0.0;
  last_imu_stamp_ = t;

  if (dt > 0.0) ahrs_update(dt, msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z, 
                            msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
}

void StickyVoNode::ahrs_update(double dt, double gx, double gy, double gz, double ax, double ay, double az) {
  const double norm = std::sqrt(ax*ax + ay*ay + az*az);
  if (norm > 1e-6) {
    const double axn = ax/norm, ayn = ay/norm, azn = az/norm;
    const double vx = 2.0*(qx_*qz_ - qw_*qy_), vy = 2.0*(qw_*qx_ + qy_*qz_), vz = qw_*qw_ - qx_*qx_ - qy_*qy_ + qz_*qz_;
    const double ex = (ayn*vz - azn*vy), ey = (azn*vx - axn*vz), ez = (axn*vy - ayn*vx);
    ex_int_ += ex*dt; ey_int_ += ey*dt; ez_int_ += ez*dt;
    gx += ahrs_kp_*ex + ahrs_ki_*ex_int_; gy += ahrs_kp_*ey + ahrs_ki_*ey_int_; gz += ahrs_kp_*ez + ahrs_ki_*ez_int_;
  }
  const double h = 0.5*dt;
  double qw=qw_, qx=qx_, qy=qy_, qz=qz_;
  qw_ += (-qx*gx - qy*gy - qz*gz)*h; qx_ += (qw*gx + qy*gz - qz*gy)*h; qy_ += (qw*gy - qx*gz + qz*gx)*h; qz_ += (qw*gz + qx*gy - qy*gx)*h;
  const double qn = std::sqrt(qw_*qw_ + qx_*qx_ + qy_*qy_ + qz_*qz_);
  if (qn > 1e-12) { qw_ /= qn; qx_ /= qn; qy_ /= qn; qz_ /= qn; }
}

} // namespace stickyvo_ros
