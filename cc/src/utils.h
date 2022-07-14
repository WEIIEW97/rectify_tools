#ifndef UTILS_H_
#define UTILS_H_

#include <math.h>
#include <time.h>

#include <Eigen/Dense>
#include <algorithm>  // for copy
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <unordered_set>
#include <vector>

#define SIGN_BIT        0x80000000
#define SIGN_BIT_IGNORE 0x7fffffff

enum CFmt { YUV444toYUV420, YUV420toYUV444 };

enum SFmt { YUV444, YUV420 };

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

// void get_yuv(const std::string& yuv_file, int width, int height, cv::Mat& y,
//              cv::Mat& u, cv::Mat& v);

unsigned char* readyuv(std::string in_path, int w, int h, SFmt fmt);

void writeyuv(std::string out_path, unsigned char* s, int w, int h, SFmt fmt);

void nv12_to_yuv444(unsigned char* in, unsigned char* out, int w, int h);

void yuv444_to_nv12(unsigned char* in, unsigned char* out, int w, int h);

unsigned char* convert_yuv(unsigned char* inbuf, int w, int h, CFmt cf);

cv::Mat yuv2mat(unsigned char* inbuf, int w, int h, SFmt fmt);

unsigned char* mat2yuv(cv::Mat src, SFmt fmt);

std::string type2str(int type);

void write_csv(std::string file, cv::Mat m);

void show_img(cv::Mat img, std::string win_name);

cv::Mat show_bilinear_img(int row, int col, const std::vector<int>& unrect_idx);

std::vector<int> get_unrect_idx(int row, int col,
                                const std::vector<int>& rect_idx);

#endif  // UTILS_H_
