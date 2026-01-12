#include <rclcpp/rclcpp.hpp>
#include "stickyvo_ros/node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<stickyvo_ros::StickyVoNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
