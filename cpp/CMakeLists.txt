cmake_minimum_required(VERSION 3.21)
project(cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/")
include_directories("/usr/local/Cellar/boost/1.76.0/include")


link_directories("/usr/local/Cellar/boost/1.76.0/lib")
add_executable(cpp
src/utils.h
src/utils.cpp
src/rectImg.h
src/rectImg.cpp
src/lutParser.h
src/lutParser.cpp
src/useLut.cpp)

target_link_libraries(cpp ${OpenCV_LIBS})