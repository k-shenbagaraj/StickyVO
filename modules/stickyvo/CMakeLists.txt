cmake_minimum_required(VERSION 3.16)
project(stickyvo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Local bundle adjustment 
find_package(Ceres REQUIRED)

add_library(stickyvo
  src/core.cpp
  src/track_db.cpp
  src/triangulation.cpp
  src/pnp.cpp
  src/local_refine.cpp
  src/bundle_adjustment.cpp
  src/vo_frontend.cpp
)

target_include_directories(stickyvo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(stickyvo PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

if (Ceres_FOUND)
  target_compile_definitions(stickyvo PUBLIC STICKYVO_HAS_CERES=1)
  target_link_libraries(stickyvo PUBLIC Ceres::ceres)
endif()

install(TARGETS stickyvo
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
