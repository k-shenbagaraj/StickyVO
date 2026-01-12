cmake_minimum_required(VERSION 3.16)
project(stickyvo_lgs)

find_package(ament_cmake REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development)

find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)

# Build the real library (outside ros2/)
add_subdirectory(
  ${CMAKE_CURRENT_LIST_DIR}/../../modules/lightgluestick
  ${CMAKE_CURRENT_BINARY_DIR}/lgs_build
)

# Install/export library target
install(TARGETS stickyvo_lgs
  EXPORT stickyvo_lgsTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(EXPORT stickyvo_lgsTargets
  FILE stickyvo_lgsTargets.cmake
  NAMESPACE stickyvo_lgs::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR}/../../modules/lightgluestick/include/
  DESTINATION include
)

install(DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR}/../../modules/lightgluestick/python/
  DESTINATION share/${PROJECT_NAME}/python
)

configure_file(
  env-hooks/50.stickyvo_lgs_pythonpath.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/50.stickyvo_lgs_pythonpath.sh
  @ONLY
)
ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/50.stickyvo_lgs_pythonpath.sh")

ament_export_include_directories(include)
ament_export_targets(stickyvo_lgsTargets NAMESPACE stickyvo_lgs:: HAS_LIBRARY_TARGET)

# Export dependencies so downstream has Python3::Python and pybind11 targets
ament_export_dependencies(Python3 pybind11_vendor pybind11)

ament_package()
