#pragma once
#include <iostream>
#include <stdio.h>
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

std::string dec2bin(int in_num, int bits_num);

int bin2dec(std::string bin_str);

cv::Mat sparse2dense(int row, int col, cv::Mat sparseMat, cv::Mat sampleX, cv::Mat sampleY);