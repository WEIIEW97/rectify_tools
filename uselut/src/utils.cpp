/**
 * @file utils.cpp
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "utils.h"

Eigen::MatrixXd normc(Eigen::MatrixXd x) {
  Eigen::MatrixXd _norm(x.rows(), x.cols());
  if (x.sum() / x.size() >= 0) {
    for (int i = 0; i < x.cols(); i++) {
      _norm.col(i) = x.col(i) / x.col(i).maxCoeff();
    }
  } else {
    for (int j = 0; j < x.rows(); j++) {
      for (int k = 0; k < x.cols(); k++) {
        _norm(j, k) = (x(j, k) - x.col(j).minCoeff()) /
                      (x.col(j).maxCoeff() - x.col(j).minCoeff());
      }
    }
  }
  return _norm;
}

void meshgrid(const cv::Range& xgv,
              const cv::Range& ygv,
              cv::Mat& X,
              cv::Mat& Y) {
  std::vector<int> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
  for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

  // need to transpose X
  cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
  cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

// dec2bin inplementation
std::string dec2bin(int in_num, int bits_num) {
  std::string bin_str = "";
  int i = 0;
  while (in_num != 0) {
    bin_str = (in_num % 2 == 0 ? "0" : "1") + bin_str;
    in_num /= 2;
    i++;
  }
  while (i < bits_num) {
    bin_str = "0" + bin_str;
    i++;
  }
  return bin_str;
}

std::string dec2bin_int_eleven(long long in_num) {
  std::string bin_str;
  bin_str = std::bitset<32>(in_num).to_string();
  return bin_str;
}

std::string dec2bin_int_nine(long long in_num) {
  std::string bin_str;
  bin_str = std::bitset<28>(in_num).to_string();
  return bin_str;
}

std::string dec2bin_int_ten(long long in_num) {
  std::string bin_str;
  bin_str = std::bitset<30>(in_num).to_string();
  return bin_str;
}

std::string dec2bin_int_eight(long long in_num) {
  std::string bin_str;
  bin_str = std::bitset<26>(in_num).to_string();
  return bin_str;
}

// convert binary string to decimal
int bin2dec(std::string bin_str) {
  int dec_num = 0;
  for (int i = 0; i < bin_str.length(); i++) {
    dec_num += (bin_str[i] - '0') * pow(2, bin_str.length() - i - 1);
  }
  return dec_num;
}

int32_t double2fixed(double num, int frac_len) {
  int32_t res;
  int _buffer;
  int int_part = 2 << (frac_len - 1);

  if (num > 0) {
    // after tons of experiments, should use flooring when input is positive
    // use ceiling when input is negative.
    _buffer = std::floor(num * int_part);
    res = (_buffer == int32_t(_buffer)) ? _buffer
                                        : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
  } else if (num < 0) {
    num = -num;
    _buffer = std::ceil(num * int_part);
    res = (_buffer == int32_t(_buffer)) ? _buffer
                                        : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
    res = res ^ SIGN_BIT;
  } else {
    res = 0;
  }
  return res;
}

double fixed2double(int32_t num, int frac_len) {
  int sign_flag = num & SIGN_BIT;  // 1 -> negative, 0 -> positive
  double res, _buffer;
  int32_t _buffer1;
  int int_part = 2 << (frac_len - 1);

  if (sign_flag == 0) {
    res = double(num) / int_part;
  } else {
    _buffer1 = num ^ SIGN_BIT;
    _buffer = double(_buffer1);
    res = _buffer / int_part;
    res = -res;
  }
  return res;
}

int sub2ind(int w, int h, int rows, int cols) {
  int ind = rows + cols * w;
  return ind;
}

int sub2ind_along_y(int w, int h, int rows, int cols) {
  int ind = cols + rows * h;
  return ind;
}

/*
usage: auto [w, h] = in2sub(...)
*/
std::tuple<int, int> ind2sub(int w, int h, int ind) {
  int rows = ind % h;
  int cols = ind / h;
  return std::make_tuple(rows, cols);
}

std::tuple<int, int> ind2sub_along_y(int w, int h, int ind) {
  int rows = ind / w;
  int cols = ind % w;
  return std::make_tuple(rows, cols);
}

// extract y, u, v channels respectivily from a 3-channel image
// void get_yuv(const std::string& yuv_file, int width, int height, cv::Mat& y,
//              cv::Mat& u, cv::Mat& v) {
//     std::ifstream file(yuv_file, std::ios::binary);
//     if (!file.is_open()) {
//         std::cout << "Cannot open file: " << yuv_file << std::endl;
//         exit(1);
//     }
//     y.create(height, width, CV_8UC1);
//     u.create(height / 2, width / 2, CV_8UC1);
//     v.create(height / 2, width / 2, CV_8UC1);
//     file.read((char*)y.data, y.total() * y.elemSize());
//     file.read((char*)u.data, u.total() * u.elemSize());
//     file.read((char*)v.data, v.total() * v.elemSize());
//     file.close();
// }

// read yuv file
unsigned char* readyuv(std::string in_path, int w, int h, SFmt fmt) {
  FILE* fp;
  if (NULL == (fp = fopen(in_path.c_str(), "rb"))) {
    printf("Cannot open file: %s\n", in_path.c_str());
    fclose(fp);
    return nullptr;
  }
  int buf_len = 0;
  if (fmt == YUV444) {
    buf_len = w * h * 3;
  } else if (fmt == YUV420) {
    buf_len = w * h * 3 / 2;
  }
  unsigned char* yuv_buf = new unsigned char[buf_len];
  fread(yuv_buf, buf_len * sizeof(unsigned char), 1, fp);
  fclose(fp);
  return yuv_buf;
}

void writeyuv(std::string out_path, unsigned char* s, int w, int h, SFmt fmt) {
  FILE* fp;
  if (nullptr == (fp = fopen(out_path.c_str(), "wb"))) {
    printf("Cannot open file: %s\n", out_path.c_str());
    fclose(fp);
    return;
  }

  int buf_len = 0;
  if (fmt == YUV444) {
    buf_len = w * h * 3;
  } else if (fmt == YUV420) {
    buf_len = w * h * 3 / 2;
  }
  fwrite(s, buf_len * sizeof(unsigned char), 1, fp);
  fclose(fp);
}

void nv12_to_yuv444(unsigned char* in, unsigned char* out, int w, int h) {
  unsigned char *srcy = nullptr, *srcu = nullptr, *srcv = nullptr;
  unsigned char *dsty = nullptr, *dstu = nullptr, *dstv = nullptr;
  srcy = in;
  srcu = srcy + w * h;
  srcv = srcu + w * h / 4;

  dsty = out;
  dstu = dsty + w * h;
  dstv = dstu + w * h;
  memcpy(dsty, srcy, w * h * sizeof(unsigned char));

  size_t i, j;
  for (i = 0; i < h; i += 2) {
    for (j = 0; j < w; j += 2) {
      unsigned char s2du = srcu[i / 2 * w / 2 + j / 2];
      dstu[i * w + j] = s2du;
      dstu[i * w + j + 1] = s2du;
      dstu[(i + 1) * w + j] = s2du;
      dstu[(i + 1) * w + j + 1] = s2du;

      unsigned char s2dv = srcv[i / 2 * w / 2 + j / 2];
      dstv[i * w + j] = s2dv;
      dstv[i * w + j + 1] = s2dv;
      dstv[(i + 1) * w + j] = s2dv;
      dstv[(i + 1) * w + j + 1] = s2dv;
    }
  }
}

void yuv444_to_nv12(unsigned char* in, unsigned char* out, int w, int h) {
  unsigned char *srcy = nullptr, *srcu = nullptr, *srcv = nullptr;
  unsigned char *dsty = nullptr, *dstuv = nullptr;
  srcy = in;
  srcu = srcy + w * h;
  srcv = srcu + w * h;

  dsty = out;
  dstuv = dsty + w * h;
  // dstv = dst + w * h / 2;

  // int w_half = w / 2;
  // int h_half = h / 2;
  memcpy(dsty, srcy, w * h * sizeof(unsigned char));

  size_t i, j;
  int idx = 0;
  for (i = 0; i < h; i += 2) {
    for (j = 0; j < w; j += 2) {
      dstuv[idx] = srcu[i * w + j];
      idx++;
      dstuv[idx] = srcv[i * w + j];
      idx++;
    }
  }
}

unsigned char* convert_yuv(unsigned char* inbuf, int w, int h, CFmt cf) {
  int buf_len = 0;
  if (cf == YUV420toYUV444) {
    buf_len = w * h * 3;
    unsigned char* p_dst = new unsigned char[buf_len];
    nv12_to_yuv444(inbuf, p_dst, w, h);
    return p_dst;
  } else {
    buf_len = w * h * 3 / 2;
    unsigned char* p_dst = new unsigned char[buf_len];
    yuv444_to_nv12(inbuf, p_dst, w, h);
    return p_dst;
  }
}

cv::Mat yuv2mat(unsigned char* inbuf, int w, int h, SFmt fmt) {
  if (fmt == YUV444) {
    cv::Mat dst = cv::Mat::zeros(h, w, CV_8UC3);
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);
    channels.at(0).data = (unsigned char*)inbuf;
    channels.at(1).data = (unsigned char*)inbuf + w * h;
    channels.at(2).data = (unsigned char*)inbuf + w * h * 2;
    cv::merge(channels, dst);
    return dst;
  } else {
    int buf_len = w * h * 3 / 2;
    cv::Mat dst = cv::Mat::zeros(h * 3 / 2, w, CV_8UC1);
    memcpy(dst.data, inbuf, buf_len * sizeof(unsigned char));
    return dst;
  }
}

unsigned char* mat2yuv(cv::Mat src, SFmt fmt) {
  int w = src.cols;
  int h = src.rows;
  if (fmt == YUV444) {
    int buf_len = w * h * 3;
    int buf_c = w * h;
    unsigned char* p_dst = new unsigned char[buf_len];
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    memcpy(p_dst, channels.at(0).data, buf_c * sizeof(unsigned char));
    memcpy(p_dst + w * h, channels.at(1).data, buf_c * sizeof(unsigned char));
    memcpy(
        p_dst + w * h * 2, channels.at(2).data, buf_c * sizeof(unsigned char));
    return p_dst;
  } else {
    int buf_len = src.total();
    unsigned char* p_dst = new unsigned char[buf_len];
    memcpy(p_dst, src.data, buf_len * sizeof(unsigned char));
    return p_dst;
  }
}

// get opencv mat type name
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

// write cv::Mat to .csv
void write_csv(std::string file, cv::Mat m) {
  std::ofstream f;
  f.open(file.c_str());
  f << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
  f.close();
}

void show_img(cv::Mat img, std::string win_name) {
  cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(win_name, img);
  cv::waitKey(0);
  cv::destroyWindow(win_name);
}

cv::Mat show_bilinear_img(int row,
                          int col,
                          const std::vector<int>& unrect_idx) {
  cv::Mat bi_linear;
  bi_linear = cv::Mat::ones(row, col, CV_8UC1) * 255;
  for (auto i : unrect_idx) {
    bi_linear.at<uint8_t>(i) = 0;
  }
  // show_img(bi_linear, win_name);
  return bi_linear;
}

std::vector<int> get_unrect_idx(int row,
                                int col,
                                const std::vector<int>& rect_idx) {
  std::unordered_set<int> rect_set;
  std::vector<int> img_idx, unrect_idx;
  for (int i = 0; i < row * col; i++) {
    img_idx.emplace_back(i);
  }

  for (int j : rect_idx) {
    rect_set.insert(j);
  }

  for (int& k : img_idx) {
    if (rect_set.find(k) == rect_set.end()) {
      unrect_idx.emplace_back(k);
    }
  }
  return unrect_idx;
}

