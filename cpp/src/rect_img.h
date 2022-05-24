#include "utils.h"

cv::Mat rect_img(const cv::Mat& xOrig2Rect, const cv::Mat& yOrig2Rect,
                 cv::Mat xRect2Orig, cv::Mat yRect2Orig, const cv::Mat& image,
                 int scale);

cv::Mat bilinear_remap(cv::Mat& img_rect, const cv::Mat& img_r,
                       const cv::Mat& img_g, const cv::Mat& img_b,
                       const cv::Mat& xy_orig_int, const cv::Mat& xy_orig_frac,
                       std::vector<int> index);

void coordinate_generator(std::vector<double>& vec, cv::Mat coeff,
                          cv::Mat img_ch, std::vector<int> index, int frac_len);