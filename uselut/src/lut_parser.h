/**
 * @file lut_parser.h
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef LUT_PARSER_H_
#define LUT_PARSER_H_

#include "utils.h"

typedef struct useLutInitParams {
  int width;
  int height;
  int int_len;
  int frac_len;
  std::string lut_path;
  std::string in_path;
  std::string out_path;
} useLutInitParams;

/**
 * @brief Initializing common function paramters.
 * 
 * @param width: Image width or image.cols. 
 * @param height: Image height or image.rows. 
 * @param int_len: Byte width of integer part. 
 * @param frac_len: Byte width of fraction part. 
 * @param lut_path: Input LUT file path. 
 * @param in_path: Input images path(or .yuv path). 
 * @param out_path: Expected rectified images output path. 
 * @return std::shared_ptr<useLutInitParams>: Pointer. 
 */
std::shared_ptr<useLutInitParams> initialize(int width,
                                             int height,
                                             int int_len,
                                             int frac_len,
                                             std::string lut_path,
                                             std::string in_path,
                                             std::string out_path);

/**
 * @brief Parsing input LUT file, returns 4 dense matrices.
 * 
 * @param params: useLutInitParams pointer. 
 * @param xOrig2Rect: Expected original to rectified dense matrix of coordinate x. Dimension (h, w /2). 
 * @param yOrig2Rect: Expected original to rectified dense matrix of coordinate y. Dimension (h, w/ 2).
 * @param xRect2Orig: Expected rectified to original dense matrix of coordinate x. Dimension (h, w).
 * @param yRect2Orig: Expected rectified to original dense matrix of coordinate y. Dimension (h, w). 
 */
void lut_parser(std::shared_ptr<useLutInitParams> params,
                cv::Mat& xOrig2Rect,
                cv::Mat& yOrig2Rect,
                cv::Mat& xRect2Orig,
                cv::Mat& yRect2Orig);

/**
 * @brief Bi-linear interpolation implementation.
 * 
 * @param row: Image height. 
 * @param col: Image width. 
 * @param sparseMat: Input sparse matrix.  
 * @param sampleX: Sample points of coordinate x. 
 * @param sampleY: Sample points of coordinate y. 
 * @param params: useLutInitParams pointer. 
 * @return cv::Mat 
 */
cv::Mat sparse2dense(int row,
                     int col,
                     cv::Mat sparseMat,
                     cv::Mat sampleX,
                     cv::Mat sampleY,
                     std::shared_ptr<useLutInitParams> params);

/**
 * @brief Converting float point number to fixed point number.
 * 
 * @param num: Input float point number. 
 * @param frac_len: Offset width. 
 * @return double 
 */
double num2fix(double num, int frac_len);

/**
 * @brief COnverting float point number to fixed point number.(without signed bit)
 * 
 * @param num 
 * @param frac_len 
 * @return double 
 */
double num2fix_unsigned(double num, int frac_len);

#endif  // LUT_PARSER_H_