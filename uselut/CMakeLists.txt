# cmake_minimum_required(VERSION 3.2)
# project(cpp)

# set(CMAKE_CXX_STANDARD 17)


# include_directories(src)


# # find_package(OpenCV REQUIRED)

# include_directories("C:/3rd/opencv/build/install/include")
# link_directories("C:/3rd/opencv/build/install/x64/vc14/lib")
# include_directories("C:/3rd/eigen/build/Eigen3/include/eigen3")
# # include_directories("C:/boost_1_79_0/boost_1_79_0")


# # link_directories("C:/boost_1_79_0/boost_1_79_0/stage/lib")
# add_executable(cpp
# src/utils.h
# src/utils.cpp
# src/rect_img.h
# src/rect_img.cpp
# src/lut_parser.h
# src/lut_parser.cpp
# src/use_lut.h
# src/use_lut.cpp 
# run.cpp)

# target_link_libraries(cpp opencv_world3416d)

cmake_minimum_required(VERSION 3.2)
project(uselut)

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

add_executable(uselut
src/utils.h
src/utils.cpp
src/rect_img.h
src/rect_img.cpp
src/lut_parser.h
src/lut_parser.cpp
src/use_lut.h
src/use_lut.cpp 
run.cpp)

target_link_libraries(uselut ${DEFAULT_LIBS})