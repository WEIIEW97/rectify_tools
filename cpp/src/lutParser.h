#include "utils.h"

void lutParser(std::string lut_file, std::string input_image_file, int int_len, int frac_len, cv::Mat& Raw2RectDenseMapX,
                   cv::Mat& Raw2RectDenseMapY, cv::Mat& Rect2RawDenseMapX, cv::Mat& Rect2RawDenseMapY);
cv::Mat sparse2dense(int row, int col, cv::Mat sparseMat, cv::Mat sampleX, cv::Mat sampleY);
double num2fix(double num, int frac_len);