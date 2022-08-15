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
std::vector<Tp> generate_lut_1d(int total_len, const cv::Mat& src) {
  const double* val = src.ptr<double>(0);
  std::vector<Tp> res;
  Tp val_converted;
  std::string val_str;

  for (int i = 0; i < src.cols; i++) {
    if (val[i] < 0) {
      val_str = dec2bin(pow(2, total_len) + val[i] * 2, total_len);
      val_converted = cal_bin2dec(val_str);
    } else {
      val_converted = num2fix_unsigned(val[i] * 2, total_len);
    }
    res.emplace_back(val_converted);
  }
  return res;
}

/**
 * @brief Kernel of function `generate_lut_2d`.
 * 
 * @tparam Tp 
 * @param total_len 
 * @param src 
 * @return std::vector<Tp> 
 */
template <typename Tp>
std::vector<Tp> generate_lut_2d_kernel(int total_len, const cv::Mat& src) {
  const double* val = src.ptr<double>(0);
  std::vector<Tp> res;
  for (int i = 0; i < src.rows; i++) {
    Tp val_converted;
    if (val[i] < 0) {
      std::string val_str = dec2bin(pow(2, total_len) + val[i], total_len);
      val_converted = cal_bin2dec(val_str);
    } else {
      // std::string val_str = dec2bin(val[i], total_len);
      // val_converted = cal_bin2dec(val_str);
      val_converted = num2fix_unsigned(val[i], total_len);
    }
    res.emplace_back(val_converted);
  }
  return res;
}

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
std::vector<Tp> generate_lut_2d(int int_len, int frac_len, const cv::Mat& src) {
  int total_len = int_len + frac_len;
  std::vector<Tp> res;
  Tp val_converted;

  cv::Mat tmp;
  for (int i = 0; i < src.cols; i++) {
    tmp.push_back(src.colRange(i, i + 1));
  }
  tmp = tmp * pow(2, frac_len);

  res = generate_lut_2d_kernel<Tp>(total_len, tmp);
  return res;
}

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
                        int total_len) {
  std::string y_str, x_str;
  std::vector<Tp> res;
  Tp val_;
  for (int i = 0; i < src_x.size(); i++) {
    y_str = dec2bin((double)src_y[i], total_len);
    x_str = dec2bin((double)src_x[i], total_len);
    y_str += x_str;
    val_ = cal_bin2dec(y_str);
    res.emplace_back(val_);
  }
  return res;
}

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
                           cv::Mat rotMat) {
  int w, h;
  // if not reversing mapping, sample width is half-size.

  w = params->img_size[1];
  h = params->img_size[0];

  int int_len = params->int_len, frac_len = params->frac_len;
  cv::Mat sample_x1, sample_x2, sample_y1, sample_y2;
  genSamplePoint(w, h, sample_x1, sample_y1, sample_x2, sample_y2);

  std::vector<Tp> orig2rect_sample_x, orig2rect_sample_y, rect2orig_sample_x,
      rect2orig_sample_y;

  // for x coord should add signed postion.
  orig2rect_sample_x = generate_lut_1d<Tp>(int_len + frac_len + 1, sample_x1);
  orig2rect_sample_y = generate_lut_1d<Tp>(int_len + frac_len, sample_y1);

  rect2orig_sample_x = generate_lut_1d<Tp>(int_len + frac_len + 1, sample_x2);
  rect2orig_sample_y = generate_lut_1d<Tp>(int_len + frac_len, sample_y2);

  matGroup orig2rect_grp, rect2orig_grp;
  orig2rect_grp = compute_delta(
      intrMatOld, kc, intrMatNew, rotMat, sample_x1, sample_y1, params, true);
  rect2orig_grp = compute_delta(
      intrMatOld, kc, intrMatNew, rotMat, sample_x2, sample_y2, params, false);

  std::vector<Tp> orig2rect_vec1, orig2rect_vec2, orig2rect_vec3;
  // generate y sample
  orig2rect_vec1 = generate_lut_2d<Tp>(int_len, frac_len, orig2rect_grp.maty);
  // generate x sample
  orig2rect_vec2 = generate_lut_2d<Tp>(int_len, frac_len, orig2rect_grp.matx);
  // pair x, y to (y, x) in one line
  orig2rect_vec3 = pair_xy(orig2rect_vec2, orig2rect_vec1, int_len + frac_len);

  std::vector<Tp> rect2orig_vec1, rect2orig_vec2, rect2orig_vec3;
  rect2orig_vec1 =
      generate_lut_2d<Tp>(int_len - 1, frac_len, rect2orig_grp.maty);
  rect2orig_vec2 =
      generate_lut_2d<Tp>(int_len - 1, frac_len, rect2orig_grp.matx);
  rect2orig_vec3 =
      pair_xy(rect2orig_vec2, rect2orig_vec1, int_len + frac_len - 1);

  std::vector<Tp> lut;

  lut.reserve(HEADER_PART_LEN + orig2rect_sample_x.size() +
              orig2rect_sample_y.size() + rect2orig_sample_x.size() +
              rect2orig_sample_y.size() + orig2rect_vec3.size() +
              rect2orig_vec3.size());

  // some fuckin fixed header info
  lut.emplace_back(163);
  lut.emplace_back(params->img_size[1]);
  lut.emplace_back(params->img_size[0]);
  lut.emplace_back((Tp)sample_y2.cols);
  lut.emplace_back((Tp)sample_x2.cols);
  lut.emplace_back((Tp)sample_y1.cols);
  lut.emplace_back((Tp)sample_x1.cols);
  lut.emplace_back(params->upcrop[1]);
  lut.emplace_back(params->upcrop[0]);
  lut.emplace_back(params->downcrop[1]);
  lut.emplace_back(params->downcrop[0]);
  lut.emplace_back(HEADER_PART_END);
  lut.emplace_back((Tp)(orig2rect_sample_x.size() + orig2rect_sample_y.size() +
                        rect2orig_sample_x.size() + rect2orig_sample_y.size() +
                        orig2rect_vec3.size() + rect2orig_vec3.size()));
  lut.emplace_back(BODY_PART_BEGIN);
  EMPLACE_BACK(orig2rect_sample_x, lut);
  EMPLACE_BACK(orig2rect_sample_y, lut);
  EMPLACE_BACK(rect2orig_sample_x, lut);
  EMPLACE_BACK(rect2orig_sample_y, lut);
  EMPLACE_BACK(orig2rect_vec3, lut);
  EMPLACE_BACK(rect2orig_vec3, lut);

  return lut;
}


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
                         std::shared_ptr<camParams> cam_params) {
  int height = params->img_size[0];
  int width = params->img_size[1];
  int image_size_t[2];
  int cheat;
  float scale;
  stereoCam<Tp> con;
  std::vector<Tp> left_, right_;
  if (width > 1920) {
    cheat = 1;
    scale = width / 1920;
    image_size_t[0] = 1080;
    image_size_t[1] = 1920;
  } else {
    cheat = 0;
    scale = 1.0f;
    image_size_t[0] = height;
    image_size_t[1] = width;
  }

  cv::Mat intrMatOldLN, intrMatOldRN, intrMatNewLN, intrMatNewRN;
  if (cheat) {
    intrMatOldLN = cam_params->intrMatOldL.clone();
    intrMatOldRN = cam_params->intrMatOldR.clone();
    intrMatNewLN = cam_params->intrMatNewL.clone();
    intrMatNewRN = cam_params->intrMatNewR.clone();
    for (int row = 0; row < cam_params->intrMatOldL.rows; row++) {
      double* old_l_p = intrMatOldLN.ptr<double>(row);
      double* old_r_p = intrMatOldRN.ptr<double>(row);
      double* new_l_p = intrMatNewLN.ptr<double>(row);
      double* new_r_p = intrMatNewRN.ptr<double>(row);
      for (int col = 0; col < cam_params->intrMatOldL.cols; col++) {
        old_l_p[col] = old_l_p[col] / scale;
        old_r_p[col] = old_r_p[col] / scale;
        new_l_p[col] = new_l_p[col] / scale;
        new_r_p[col] = new_r_p[col] / scale;
      }
    }
    intrMatOldLN.ptr<double>(2)[2] = 1;
    intrMatOldRN.ptr<double>(2)[2] = 1;
    intrMatNewLN.ptr<double>(2)[2] = 1;
    intrMatNewRN.ptr<double>(2)[2] = 1;
    // left cam
    left_ = genlut_run<Tp>(params,
                           intrMatOldLN,
                           cam_params->kcL,
                           intrMatNewLN,
                           cam_params->rotMatL);
    // right cam
    right_ = genlut_run<Tp>(params,
                            intrMatOldRN,
                            cam_params->kcR,
                            intrMatNewRN,
                            cam_params->rotMatR);

  } else {
    // left cam
    left_ = genlut_run<Tp>(params,
                           cam_params->intrMatOldL,
                           cam_params->kcL,
                           cam_params->intrMatNewL,
                           cam_params->rotMatL);
    // right cam
    right_ = genlut_run<Tp>(params,
                            cam_params->intrMatOldR,
                            cam_params->kcR,
                            cam_params->intrMatNewR,
                            cam_params->rotMatR);

    con.left_cam = left_;
    con.right_cam = right_;
  }
  return con;
}

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
                  char const* which_cam) {
  char tr[300];
  int i;
  FILE* stream;
  int width = params->img_size[1];
  int height = params->img_size[0];
  char w[10], h[10];
#ifdef _WIN32
  sprintf_s(w, "%d", width);
  sprintf_s(h, "%d", height);

  /**
   * @brief Writing .txt file.
   * 
   */
  strcpy_s(tr, path);
  strcat_s(tr, "LutDec");  //LutDec
  strcat_s(tr, whichCam);
  strcat_s(tr, "_");     //_
  strcat_s(tr, w);       //w
  strcat_s(tr, "_");     //_
  strcat_s(tr, h);       //h
  strcat_s(tr, ".txt");  //.txt
  fopen_s(&stream, tr, "w");
  for (auto i : tbdata) {
    fprintf(stream, "lld\n", i);
  }
  fclose(stream);

  /**
   * @brief Writing .bin file.
   * 
   */
  std::vector<Tp> bin_data;
  bin_data = Make_Blendian_Data(tbdata);
  strcpy_s(tr, path);
  strcat_s(tr, "LutBin");  //LutBin
  strcat_s(tr, whichCam);
  strcat_s(tr, "_");     //_
  strcat_s(tr, w);       //w
  strcat_s(tr, "_");     //_
  strcat_s(tr, h);       //h
  strcat_s(tr, ".bin");  //.bin
  fopen_s(&stream, tr, "wb");
  fwrite(bin_data.data(), 4, bin_data.size(), stream);
  fclose(stream);
#else
  sprintf(w, "%d", width);
  sprintf(h, "%d", height);

  /**
   * @brief Writing .txt file.
   * 
   */
  strcpy(tr, path);
  strcat(tr, "LutDec");  //LutDec
  strcat(tr, which_cam);
  strcat(tr, "_");     //_
  strcat(tr, w);       //w
  strcat(tr, "_");     //_
  strcat(tr, h);       //h
  strcat(tr, ".txt");  //.txt
  stream = fopen(tr, "w");
  for (auto i : tbdata) {
    fprintf(stream, "%lld\n", i);
  }
  fclose(stream);

  /**
   * @brief Writing .bin file.
   * 
   */
  std::vector<Tp> bin_data;
  bin_data = Make_BLendian_Data(tbdata);
  strcpy(tr, path);
  strcat(tr, "LutBin");  //LutBin
  strcat(tr, which_cam);
  strcat(tr, "_");     //_
  strcat(tr, w);       //w
  strcat(tr, "_");     //_
  strcat(tr, h);       //h
  strcat(tr, ".bin");  //.bin
  stream = fopen(tr, "wb");
  fwrite(bin_data.data(), 4, bin_data.size(), stream);
  fclose(stream);
#endif
}

template <typename Tp>
int NVP_cam_writefile(std::shared_ptr<genLutInitParams> params,
                      char const* path,
                      bool is_mono,
                      char const* name_caml,
                      char const* name_camr,
                      stereoCam<Tp> container) {
  int h = params->img_size[0];
  int w = params->img_size[1];

  if (w > 1920) return Imgsize_error;

  std::vector<Tp> cam_left;
  std::vector<Tp> cam_right;

  cam_left = container.left_cam;
  cam_right = container.right_cam;

  if (is_mono) {
    write_txtbin(params, cam_left, path, name_caml);
  } else {
    write_txtbin(params, cam_left, path, name_caml);
    write_txtbin(params, cam_left, path, name_camr);
  }

  return Nvp_Success;
}
#endif  // GENERATE_DATA_H