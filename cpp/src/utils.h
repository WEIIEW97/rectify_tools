#include <iostream>
#include <fstream>
#include <cstdio>
#include <tuple>
#include <cmath>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>


#define SIGN_BIT         0x80000000
#define SIGN_BIT_IGNORE  0x7fffffff

Eigen::MatrixXd normc(Eigen::MatrixXd x);

// meshgrid method implemented by cv::repeat
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

std::string dec2bin(int in_num, int bits_num);

int bin2dec(std::string bin_str);

int32_t double2fixed(double num, int frac_len);

double fixed2double(int32_t num, int frac_len);

int sub2ind(int w, int h, int rows, int cols);

std::tuple<int, int> ind2sub(int w, int h, int ind);

bool ismember(double num, std::vector<double> vec);

void get_yuv(const std::string& yuv_file, int width, int height, cv::Mat& y, cv::Mat& u, cv::Mat& v);