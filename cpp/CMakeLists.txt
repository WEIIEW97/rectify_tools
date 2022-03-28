cmake_minimum_required(VERSION 3.21)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/")
add_executable(cpp
src/useLut.cpp
src/utils.cpp
src/utils.h
src/rectimg.h
src/rectimg.cpp)

target_link_libraries(cpp ${OpenCV_LIBS})