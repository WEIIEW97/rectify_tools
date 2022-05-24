#include <Eigen/Dense>
#include <algorithm>  // for copy
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

#define SIGN_BIT 0x80000000
#define SIGN_BIT_IGNORE 0x7fffffff

Eigen::MatrixXd normc(Eigen::MatrixXd x);

// meshgrid method implemented by cv::repeat
void meshgrid(const cv::Range& xgv, const cv::Range& ygv, cv::Mat& X,
              cv::Mat& Y);

std::string dec2bin(int in_num, int bits_num);

int bin2dec(std::string bin_str);

int32_t double2fixed(double num, int frac_len);

double fixed2double(int32_t num, int frac_len);

int sub2ind(int w, int h, int rows, int cols);

int sub2ind_along_y(int w, int h, int rows, int cols);

std::tuple<int, int> ind2sub(int w, int h, int ind);

std::tuple<int, int> ind2sub_along_y(int w, int h, int ind);

bool ismember(double num, std::vector<double> vec);

void get_yuv(const std::string& yuv_file, int width, int height, cv::Mat& y,
             cv::Mat& u, cv::Mat& v);

std::string type2str(int type);

void write_csv(std::string file, cv::Mat m);