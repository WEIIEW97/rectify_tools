cmake_minimum_required(VERSION 3.21)
project(cc)

set(CMAKE_CXX_STANDARD 17)
# open release mode o3 optimization
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -D_DEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

include_directories(src)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/")
#include_directories("/usr/local/Cellar/boost/1.78.0_1/include")
#
#
#link_directories("/usr/local/Cellar/boost/1.78.0_1/lib")

add_executable(cc
src/utils.h
src/utils.cc
src/rect_img.h
src/rect_img.cc
src/lut_parser.h
src/lut_parser.cc
src/use_lut.h
src/use_lut.cc
src/gen_lut.h
src/gen_lut.cc
run.cc
)

target_link_libraries(cc ${OpenCV_LIBS})
