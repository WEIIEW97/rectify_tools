#include "utils.h"

// function paramters data type
// function imgRect(xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig, image, KK_new, Kinit, k, R, imgOrig, paraDir, whichOne, scale)



cv::Mat RectImg(cv::Mat xOrig2Rect, cv::Mat yOrig2Rect, cv::Mat xRect2Orig, cv::Mat yRect2Orig, cv::Mat image, cv::Mat KK_new,
cv::Mat Kinit, cv::Mat K, cv::Mat R, cv::Mat imgOrig, std::string paraDir, std::string whichOne, int scale);