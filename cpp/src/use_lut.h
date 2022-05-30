#ifndef USE_LUT_H_
#define USE_LUT_H_

#include "lut_parser.h"
#include "rect_img.h"
#include "utils.h"

#define BILINEAR_IMG_WINNAME "bilinear_img"
#define RECT_IMG_WINNAME "rect_img"

void use_lut(int row, int col, int int_len, int frac_len,
             const std::string& lut_file, const std::string& yuv_path,
             const std::string& out_path, bool show_bi_img);

#endif  // USE_LUT_H_