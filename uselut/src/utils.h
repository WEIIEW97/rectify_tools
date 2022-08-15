/**
 * @file utils.h
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>

#define SIGN_BIT        0x80000000
#define SIGN_BIT_IGNORE 0x7fffffff

enum CFmt { YUV444toYUV420, YUV420toYUV444 };

enum SFmt { YUV444, YUV420 };

Eigen::MatrixXd normc(Eigen::MatrixXd x);

/**
 * @brief Meshgrid method implemented by cv::repeat.
 * 
 * @param xgv 
 * @param ygv 
 * @param X 
 * @param Y 
 */
void meshgrid(const cv::Range& xgv,
              const cv::Range& ygv,
              cv::Mat& X,
              cv::Mat& Y);

/**
 * @brief Converting decimal to binary array. 
 * WARNING: Use it carefully when byte width is larger that 32.
 *          May cause data overflow error. Use `bitset` intead.
 * @param in_num: Input int number. 
 * @param bits_num: Byte width. 
 * @return std::string 
 */
std::string dec2bin(int in_num, int bits_num);

/**
 * @brief Bitset version of converting decimal to binary array with
 *        fixed integer length of 11.
 * 
 * @param in_num: Input long long number.
 * @return std::string 
 */
std::string dec2bin_int_eleven(long long in_num);

std::string dec2bin_int_nine(long long in_num);

std::string dec2bin_int_ten(long long in_num);

std::string dec2bin_int_eight(long long in_num);

int bin2dec(std::string bin_str);

int32_t double2fixed(double num, int frac_len);

double fixed2double(int32_t num, int frac_len);

int sub2ind(int w, int h, int rows, int cols);

int sub2ind_along_y(int w, int h, int rows, int cols);

std::tuple<int, int> ind2sub(int w, int h, int ind);

std::tuple<int, int> ind2sub_along_y(int w, int h, int ind);

template <typename Tp>
bool ismember(Tp num, std::vector<Tp> vec);

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

/**
 * @brief Get the unrect idx object
 * 
 * @param row: Image height. 
 * @param col: Image width. 
 * @param rect_idx: Indices of rectified pixels. 
 * @return std::vector<int> 
 */
std::vector<int> get_unrect_idx(int row,
                                int col,
                                const std::vector<int>& rect_idx);

template <typename Tp>
void save_vector_to_txt(const char* path, std::vector<Tp> src);
#endif  // UTILS_H_