#include "utils.h"

void lut_parser(const std::string& lut_file, int int_len, int frac_len,
                cv::Mat& xOrig2Rect, cv::Mat& yOrig2Rect, cv::Mat& xRect2Orig,
                cv::Mat& yRect2Orig);

cv::Mat sparse2dense(int row, int col, cv::Mat sparseMat, cv::Mat sampleX,
                     cv::Mat sampleY);

double num2fix(double num, int frac_len);

double num2fix_unsigned(double num, int frac_len);
