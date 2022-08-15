/**
 * @file generate_lut.h
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief Header file for generating LUT info.
 * @version 0.1
 * @date 2022-08-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#ifndef GENERATE_LUT_H
#define GENERATE_LUT_H

#include "../../uselut/src/lut_parser.h"
#include "../../uselut/src/utils.h"
#include "gen_sample_point.h"

#include <cstdlib>
#include <memory>

#include <opencv2/core/fast_math.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

#define SIGN_BIT        0x80000000
#define SIGN_BIT_IGNORE 0x7fffffff
#define HEADER_PART_END 0x5a5a5a5a
#define BODY_PART_BEGIN 0xffffffff
#define HEADER_PART_LEN 14

#define VECTOR_PRINTER(vec)      \
  for (auto i : vec) {           \
    std::cout << i << std::endl; \
  }

#define PUSH_BACK(src, dst) \
  for (auto i : src) {      \
    dst.push_back(i);       \
  }

#define EMPLACE_BACK(src, dst) \
  for (auto i : src) {         \
    dst.emplace_back(i);       \
  }

enum ERROR_FLAG {
  Nvp_Success = 0,
  CamLen_error = -1,
  CamLRLen_error = -2,
  XmlPathName_error = -3,
  XmlRoot_error = -4,
  Imgsize_error = -5,
};

typedef struct genLutInitParams {
  int img_size[2];
  int upcrop[2];
  int downcrop[2];
  int int_len;
  int frac_len;
} genLutInitParams;

typedef struct matGroup {
  cv::Mat maty;
  cv::Mat matx;
} matGroup;

typedef struct camParams {
  cv::Mat transVec;
  cv::Mat intrMatOldL;
  cv::Mat intrMatOldR;
  cv::Mat kcL;
  cv::Mat kcR;
  cv::Mat intrMatNewL;
  cv::Mat intrMatNewR;
  cv::Mat rotMatL;
  cv::Mat rotMatR;
} camPrams;

template <typename Tp>
struct stereoCam {
  std::vector<Tp> left_cam;
  std::vector<Tp> right_cam;
};

/**
 * @brief Initialzing camera parameters, return a pointer.
 * 
 * @param transVecT:Transformation matrix. 
 * @param intrMatOldLT: Old left camera intrinsic matrix.
 * @param intrMatOldRT: Old right camera intrinsic matrix.
 * @param kcLT: Left camera distortion matrix.  
 * @param kcRT: Right camera distortion matrix.
 * @param intrMatNewLT: New left camera intrinsic matrix.
 * @param intrMatNewRT: New right camera intrinsic matrix.
 * @param rotMatLT: Left rotation matrix. 
 * @param rotMatRT: Right rotation matrix. 
 * @return std::shared_ptr<camParams> 
 */
std::shared_ptr<camParams> Get_Cam_Parameter(cv::Mat transVecT,
                                             cv::Mat intrMatOldLT,
                                             cv::Mat intrMatOldRT,
                                             cv::Mat kcLT,
                                             cv::Mat kcRT,
                                             cv::Mat intrMatNewLT,
                                             cv::Mat intrMatNewRT,
                                             cv::Mat rotMatLT,
                                             cv::Mat rotMatRT);

/**
 * @brief Computing the delta LUT sparse matrix.
 * 
 * @param intrMatOld 
 * @param kc 
 * @param intrMatNew 
 * @param rotMat 
 * @param sampledX: Sample points for `X` coordinates. 
 * @param sampledY: Sample points for `Y` coordinates.
 * @param params: Params pointer. 
 * @param reverse: If reversing mapping.
 * @return matGroup 
 */
matGroup compute_delta(cv::Mat intrMatOld,
                       cv::Mat kc,
                       cv::Mat intrMatNew,
                       cv::Mat rotMat,
                       cv::Mat sampledX,
                       cv::Mat sampledY,
                       std::shared_ptr<genLutInitParams> params,
                       bool reverse);

/**
 * @brief Generating LUT content of sample x and sample y, input should be a `(1 x N)` vector. 
 * 
 * @tparam Tp: Template of expected data type.
 * @param total_len: int_len + frac_len 
 * @param src: Input vector. 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> generate_lut_1d(int total_len, const cv::Mat& src);

/**
 * @brief Kernel of function `generate_lut_2d`.
 * 
 * @tparam Tp 
 * @param total_len 
 * @param src 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> generate_lut_2d_kernel(int total_len, const cv::Mat& src);

/**
 * @brief Generating LUT content of sparse delta matrix, input should be a `(M x N)` matrix.
 * 
 * @tparam Tp: Template of expected data type.
 * @param int_len: Byte width for integer part. 
 * @param frac_len: Byte width for fraction part. 
 * @param src: Input matrix. 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> generate_lut_2d(int int_len, int frac_len, const cv::Mat& src);

/**
 * @brief Concatenating byte array y and byte array x to a new byte array yx.
 * 
 * @tparam Tp 
 * @param src_x: Input byte array x. 
 * @param src_y: Input byte array y. 
 * @param total_len: int_len + frac_len 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> pair_xy(std::vector<Tp> src_x,
                        std::vector<Tp> src_y,
                        int total_len);

/**
 * @brief Running the process of generating LUT data.
 * 
 * @tparam Tp: Expected data type.
 * @param params 
 * @param intrMatOld 
 * @param kc 
 * @param intrMatNew 
 * @param rotMat 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> genlut_run(std::shared_ptr<genLutInitParams> params,
                           cv::Mat intrMatOld,
                           cv::Mat kc,
                           cv::Mat intrMatNew,
                           cv::Mat rotMat);

template <typename Tp>
void save_vector_to_txt(const char* path, std::vector<Tp> src);

/**
 * @brief Generating LUT data for stereo system.
 * 
 * @tparam Tp 
 * @param params 
 * @param cam_params: Initial camera params pointer. 
 * @return stereoCam<Tp> 
 */
template <typename Tp>
stereoCam<Tp> stereo_run(std::shared_ptr<genLutInitParams> params,
                         std::shared_ptr<camParams> cam_params);

/**
 * @brief Writing LUT to .txt file.
 * 
 * @tparam Tp 
 * @param params 
 * @param tbdata: Input vector. 
 * @param path: Destination path. 
 * @param which_cam: Marker, 'L' or 'R'. 
 */
template <typename Tp>
void write_txtbin(std::shared_ptr<genLutInitParams> params,
                  std::vector<Tp> tbdata,
                  char const* path,
                  char const* which_cam);

template <typename Tp>
int NVP_cam_writefile(std::shared_ptr<genLutInitParams> params,
                      char const* path,
                      bool is_mono,
                      char const* name_caml,
                      char const* name_camr,
                      stereoCam<Tp> container);
#endif  // GENERATE_DATA_H