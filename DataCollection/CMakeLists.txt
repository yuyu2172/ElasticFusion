cmake_minimum_required(VERSION 2.6.0)

project(ElasticFusion)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2)
find_package(vis3d)

set (CMAKE_CXX_STANDARD 11)

add_executable(collect_data src/collect_data.cpp)
target_link_libraries(collect_data PUBLIC realsense2 vis3d)
