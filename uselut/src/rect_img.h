/**
 * @file rect_img.h
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef RECT_IMG_H_
#define RECT_IMG_H_

#include "lut_parser.h"
#include "utils.h"
// construct a struct or class to return 2 cv::Mat
typedef struct rectBuffer {
  cv::Mat rect_img;
  std::vector<int> rect_idx;
} rectBuffer;

/**
 * @brief Rectifying input image by 4 dense matrix.
 * 
 * @param xOrig2Rect 
 * @param yOrig2Rect 
 * @param xRect2Orig 
 * @param yRect2Orig 
 * @param image: Input image. 
 * @param params: useLutInitParams pointer. 
 * @return rectBuffer 
 */
rectBuffer rect_img(const cv::Mat& xOrig2Rect,
                    const cv::Mat& yOrig2Rect,
                    cv::Mat xRect2Orig,
                    cv::Mat yRect2Orig,
                    const cv::Mat& image,
                    std::shared_ptr<useLutInitParams> params);

/**
 * @brief Bi-linear interpolation implementation.
 * 
 * @param img_r: Red channel. 
 * @param img_g: Green channel. 
 * @param img_b: Blue channel. 
 * @param final_xy_orig_int: Byte width of integer part. 
 * @param final_xy_orig_frac: Byte width of fraction part. 
 * @param ind: Rectified reference indices. (aligned by column order) 
 * @param params: useLutInitParams pointer. 
 * @return cv::Mat 
 */
cv::Mat bilinear_remap(const cv::Mat& img_r,
                       const cv::Mat& img_g,
                       const cv::Mat& img_b,
                       const cv::Mat& final_xy_orig_int,
                       const cv::Mat& final_xy_orig_frac,
                       std::vector<int> ind,
                       std::shared_ptr<useLutInitParams> params);

void coordinate_generator(std::vector<double>& vec,
                          cv::Mat coeff,
                          cv::Mat img_ch,
                          std::vector<int> index,
                          int frac_len);

#endif  // RECT_IMG_H_