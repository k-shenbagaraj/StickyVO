cmake_minimum_required(VERSION 3.16)
project(stickyvo_lgs LANGUAGES CXX)

add_library(stickyvo_lgs STATIC
  src/lgs_frontend.cpp
)

target_include_directories(stickyvo_lgs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(stickyvo_lgs PUBLIC cxx_std_17)

find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# CRITICAL: PRIVATE so Python3::Python does not appear in the exported interface
target_link_libraries(stickyvo_lgs PRIVATE
  pybind11::embed
  Python3::Python
)

add_library(stickyvo_lgs::stickyvo_lgs ALIAS stickyvo_lgs)
