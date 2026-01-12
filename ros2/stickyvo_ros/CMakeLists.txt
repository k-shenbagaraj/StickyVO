cmake_minimum_required(VERSION 3.16)
project(stickyvo_ros)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development)

find_package(stickyvo_lgs REQUIRED)

add_executable(stickyvo_node
  src/main.cpp
  src/node.cpp
)

target_include_directories(stickyvo_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ament_target_dependencies(stickyvo_node
  rclcpp
  sensor_msgs
)

target_link_libraries(stickyvo_node
  stickyvo_lgs::stickyvo_lgs
  ${OpenCV_LIBS}
)

install(TARGETS stickyvo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../modules/stickyvo
                 ${CMAKE_CURRENT_BINARY_DIR}/core_build)

target_link_libraries(stickyvo_node
  stickyvo_lgs::stickyvo_lgs
  stickyvo
  ${OpenCV_LIBS}
)
target_include_directories(stickyvo_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CMAKE_CURRENT_LIST_DIR}/../../modules/stickyvo/include
)



ament_package()
