cmake_minimum_required(VERSION 3.2)
project(genlut)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
find_package(Boost 1.40 REQUIRED COMPONENTS
        filesystem
        program_options
        system
        thread)
find_package(OpenCV 3.4 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(include
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRECTORIES})
include_directories(3rd/ethzasl_apriltag2/include)

link_directories(${Boost_LIBRARY_DIRS})

add_definitions(-DHAVE_OPENCV3)
add_definitions(-DBOOST_ALL_DYN_LINK)
message(STATUS "defined HAVE_OPENCV3")

set(DEFAULT_LIBS
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_THREAD_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_executable(genlut
src/gen_sample_point.h
src/gen_sample_point.cc
src/generate_lut.h
src/generate_lut.cc
)

target_link_libraries(genlut ${DEFAULT_LIBS})