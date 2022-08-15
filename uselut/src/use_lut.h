/**
 * @file use_lut.h
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef USE_LUT_H_
#define USE_LUT_H_

#include "lut_parser.h"
#include "rect_img.h"
#include "utils.h"

#define BILINEAR_IMG_WINNAME "bilinear_img"
#define RECT_IMG_WINNAME     "rect_img"

void use_lut(std::shared_ptr<useLutInitParams> params, bool show_bi_img, bool is_rgb);

#endif  // USE_LUT_H_