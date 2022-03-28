#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>


Eigen::MatrixXd normc(Eigen::MatrixXd x);

// meshgrid method implemented by cv::repeat
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

cv::Mat comp_distortion_oulu(cv::Mat xd, cv::Mat k);

cv::Mat normalize_pixel(cv::Mat x_kk, cv::Mat fc, cv::Mat cc, cv::Mat kc, double alpha_c);

cv::Mat Orig2Rect(cv::Mat pix, cv::Mat intrMatOld, cv::Mat intrMatNew, cv::Mat R, cv::Mat kc);