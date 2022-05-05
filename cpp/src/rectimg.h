#include "utils.h"

cv::Mat rectImg(cv::Mat xOrig2Rect, const cv::Mat& yOrig2Rect, cv::Mat xRect2Orig, cv::Mat yRect2Orig, cv::Mat image, cv::Mat imgOrig, int scale);

cv::Mat bilinear_remap(cv::Mat& img_rect, cv::Mat img_r, cv::Mat img_g, cv::Mat img_b, cv::Mat xy_orig_int, cv::Mat xy_orig_frac, std::vector<int> index);