/**
 * @file use_lut.cpp
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "use_lut.h"

#include <memory>

void use_lut(std::shared_ptr<useLutInitParams> params,
             bool show_bi_img,
             bool is_rgb) {
  int row = params->height, col = params->width, int_len = params->int_len,
      frac_len = params->frac_len;
  std::string lut_file = params->lut_path, in_path = params->in_path,
              out_path = params->out_path;
  cv::Mat xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig;
  lut_parser(params, xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig);

  cv::Mat src;
  unsigned char* yuv420_buf = readyuv(in_path, col, row, SFmt::YUV420);
  unsigned char* yuv444_buf =
      convert_yuv(yuv420_buf, col, row, CFmt::YUV420toYUV444);
  src = yuv2mat(yuv444_buf, col, row, SFmt::YUV444);

  delete[] yuv420_buf;
  delete[] yuv444_buf;

  rectBuffer y_rect;
  y_rect =
      rect_img(xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig, src, params);

  unsigned char* dst_buf = mat2yuv(y_rect.rect_img, SFmt::YUV444);
  unsigned char* dst420_buf =
      convert_yuv(dst_buf, col, row, CFmt::YUV444toYUV420);

  cv::Mat rect_y;
  cv::extractChannel(y_rect.rect_img, rect_y, 0);

  std::vector<int> _param = {0};
  cv::imwrite(out_path, rect_y, _param);
  std::string yuv_path = "D:/rectify_tools/cpp/data/output/rect1920.yuv";
  writeyuv(yuv_path, dst420_buf, col, row, SFmt::YUV420);
  delete[] dst_buf;
  delete[] dst420_buf;
  std::vector<int> unrect;
  unrect = get_unrect_idx(row, col, y_rect.rect_idx);

  if (show_bi_img) {
    cv::Mat bi_img;
    bi_img = show_bilinear_img(row, col, unrect);
    show_img(bi_img, BILINEAR_IMG_WINNAME);
  }

  printf("The number of unrectified pixels: %zu\n", unrect.size());
}
