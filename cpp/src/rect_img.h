#ifndef RECT_IMG_H_
#define RECT_IMG_H_

#include "utils.h"

// construct a struct or class to return 2 cv::Mat
typedef struct rectBuffer {
    cv::Mat rect_img;
    std::vector<int> rect_idx;
} rectBuffer;

rectBuffer rect_img(const cv::Mat& xOrig2Rect, const cv::Mat& yOrig2Rect,
                    cv::Mat xRect2Orig, cv::Mat yRect2Orig,
                    const cv::Mat& image, int scale);

cv::Mat bilinear_remap(const cv::Mat& img_r, const cv::Mat& img_g,
                       const cv::Mat& img_b, const cv::Mat& xy_orig_int,
                       const cv::Mat& xy_orig_frac, std::vector<int> index);

void coordinate_generator(std::vector<double>& vec, cv::Mat coeff,
                          cv::Mat img_ch, std::vector<int> index, int frac_len);

#endif  // RECT_IMG_H_