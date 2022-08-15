/**
 * @file lut_parser.cpp
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "lut_parser.h"

#include "utils.h"

#include <memory>

std::shared_ptr<useLutInitParams> initialize(int width,
                                             int height,
                                             int int_len,
                                             int frac_len,
                                             std::string lut_path,
                                             std::string in_path,
                                             std::string out_path) {
  std::shared_ptr<useLutInitParams> params(new useLutInitParams);
  params->width = width;
  params->height = height;
  params->int_len = int_len;
  params->frac_len = frac_len;
  params->lut_path = lut_path;
  params->in_path = in_path;
  params->out_path = out_path;
  return params;
}

// read lookup table file and input image file
void lut_parser(std::shared_ptr<useLutInitParams> params,
                cv::Mat& xOrig2Rect,
                cv::Mat& yOrig2Rect,
                cv::Mat& xRect2Orig,
                cv::Mat& yRect2Orig) {
  std::vector<double> lut_info;
  const std::string lut_file = params->lut_path;
  const int int_len = params->int_len, frac_len = params->frac_len;

  std::ifstream file(lut_file);
  assert(file.is_open());
  std::string line;
  while (std::getline(file, line)) {
    lut_info.push_back(std::stod(line));
  }
  file.close();

  double col_num, row_num, max_diff;
  col_num = lut_info[1];
  row_num = lut_info[2];

  max_diff = std::max(col_num, row_num);

  double rect2raw_sample_row_num, rect2raw_sample_col_num,
      raw2rect_sample_row_num, raw2rect_sample_col_num;
  rect2raw_sample_row_num = lut_info[3];
  rect2raw_sample_col_num = lut_info[4];
  raw2rect_sample_row_num = lut_info[5];
  raw2rect_sample_col_num = lut_info[6];

  int index = 14;
  std::vector<double> raw2rectSampleX, raw2rectSampleY, rect2rawSampleX,
      rect2rawSampleY;

  for (int i = 0; i < raw2rect_sample_col_num; i++) {
    raw2rectSampleX.push_back(lut_info[index] / 2);
    index++;
  }

  for (int i = 0; i < raw2rect_sample_row_num; i++) {
    raw2rectSampleY.push_back(lut_info[index] / 2);
    index++;
  }

  const int t = int_len + frac_len;
  const double thr1 = pow(2, t), thr2 = pow(2, t - 1);

  for (double& i : raw2rectSampleX) {
    if (i > 2 * max_diff) {
      i = i - thr1;
    }
  }

  for (double& i : raw2rectSampleY) {
    if (i > 2 * max_diff) {
      i = i - thr2;
    }
  }

  for (double& i : raw2rectSampleX) {
    if (i < -max_diff) {
      i = i + thr1;
    }
  }

  for (double& i : raw2rectSampleY) {
    if (i < -max_diff) {
      i = i + thr2;
    }
  }

  for (int i = 0; i < rect2raw_sample_col_num; i++) {
    rect2rawSampleX.push_back(lut_info[index] / 2);
    index++;
  }

  for (int i = 0; i < rect2raw_sample_row_num; i++) {
    rect2rawSampleY.push_back(lut_info[index] / 2);
    index++;
  }

  for (double& i : rect2rawSampleX) {
    if (i > 2 * max_diff) {
      i = i - thr1;
    }
  }

  for (double& i : rect2rawSampleY) {
    if (i > 2 * max_diff) {
      i = i - thr2;
    }
  }

  for (double& i : rect2rawSampleX) {
    if (i < -max_diff) {
      i = i + thr1;
    }
  }

  for (double& i : rect2rawSampleY) {
    if (i < -max_diff) {
      i = i + thr2;
    }
  }

  cv::Mat raw2rectSampleX_mat(
      (int)raw2rectSampleX.size(), 1, CV_64F, raw2rectSampleX.data());
  cv::Mat raw2rectSampleY_mat(
      (int)raw2rectSampleY.size(), 1, CV_64F, raw2rectSampleY.data());
  cv::Mat rect2rawSampleX_mat(
      (int)rect2rawSampleX.size(), 1, CV_64F, rect2rawSampleX.data());
  cv::Mat rect2rawSampleY_mat(
      (int)rect2rawSampleY.size(), 1, CV_64F, rect2rawSampleY.data());

  cv::transpose(raw2rectSampleX_mat, raw2rectSampleX_mat);
  cv::transpose(rect2rawSampleX_mat, rect2rawSampleX_mat);

  cv::Mat raw2rectSample_x, raw2rectSample_y, rect2rawSample_x,
      rect2rawSample_y;
  // don't know why, but if change from `rect2raw_sample_row_num` to
  // `raw2rect_sample_row_num` then will cause an error.
  // however, they are the same.
  // update: fixed this error.
  cv::repeat(
      raw2rectSampleX_mat, (int)raw2rect_sample_row_num, 1, raw2rectSample_x);
  cv::repeat(
      raw2rectSampleY_mat, 1, (int)raw2rect_sample_col_num, raw2rectSample_y);

  cv::repeat(
      rect2rawSampleX_mat, (int)rect2raw_sample_row_num, 1, rect2rawSample_x);
  cv::repeat(
      rect2rawSampleY_mat, 1, (int)rect2raw_sample_col_num, rect2rawSample_y);

  int raw2rect_int_len = int_len;
  int raw2rect_frac_len = frac_len;
  int raw2rect_world_len = raw2rect_int_len + raw2rect_frac_len;

  std::vector<double> raw2rect_delta_sample;
  for (int i = 0; i < raw2rect_sample_row_num * raw2rect_sample_col_num; i++) {
    raw2rect_delta_sample.push_back(lut_info[index]);
    index++;
  }

  // convert to binary with raw2rect_world_len * 2bits
  std::vector<std::string> raw2rect_delta_sample_bin;

  switch (int_len) {
    case 9:
      raw2rect_delta_sample_bin.reserve(raw2rect_delta_sample.size());
      for (double i : raw2rect_delta_sample) {
        raw2rect_delta_sample_bin.push_back(dec2bin_int_nine(i));
      }
      break;
    case 11:
      raw2rect_delta_sample_bin.reserve(raw2rect_delta_sample.size());
      for (double i : raw2rect_delta_sample) {
        raw2rect_delta_sample_bin.push_back(dec2bin_int_eleven(i));
      }
      break;
    default:
      printf(
          "Undefined integer bit length. Only suppport int_len = 11 && int_len "
          "= 9 right now.");
  }

  std::vector<double> raw2rect_delta_sample_x, raw2rect_delta_sample_y;
  // convert binary to decimal for raw2rec_delta_sample_bin
  for (auto& i : raw2rect_delta_sample_bin) {
    auto _len = i.length();
    // subsrting from 0 to raw2rect_world_len in raw2rect_delta_sample_bin
    std::string _suby = i.substr(0, raw2rect_world_len);
    /* different compared to matlab, substr takes range of [a,b) */
    std::string _subx = i.substr(raw2rect_world_len, _len);
    raw2rect_delta_sample_y.push_back(bin2dec(_suby) /
                                      pow(2, raw2rect_frac_len));
    raw2rect_delta_sample_x.push_back(bin2dec(_subx) /
                                      pow(2, raw2rect_frac_len));
  }

  for (double& i : raw2rect_delta_sample_y) {
    if (i > 0.5 * pow(2, raw2rect_int_len - 1)) {
      i = i - pow(2, raw2rect_int_len);
    }
  }

  for (double& i : raw2rect_delta_sample_x) {
    if (i > 0.5 * pow(2, raw2rect_int_len - 1)) {
      i = i - pow(2, raw2rect_int_len);
    }
  }

  for (double& i : raw2rect_delta_sample_y) {
    if (i < -0.5 * pow(2, raw2rect_int_len)) {
      i = i + pow(2, raw2rect_int_len);
    }
  }

  for (double& i : raw2rect_delta_sample_x) {
    if (i < -0.5 * pow(2, raw2rect_int_len)) {
      i = i + pow(2, raw2rect_int_len);
    }
  }

  cv::Mat raw2rect_delta_samplex((int)raw2rect_sample_col_num,
                                 (int)raw2rect_sample_row_num,
                                 CV_64F,
                                 raw2rect_delta_sample_x.data());
  cv::Mat raw2rect_delta_sampley((int)raw2rect_sample_col_num,
                                 (int)raw2rect_sample_row_num,
                                 CV_64F,
                                 raw2rect_delta_sample_y.data());

  /**
     * do the same thing for rect to raw
     **/
  // get the length of int and frac
  int rect2raw_int_len = int_len - 1;  // exclude sign bit
  int rect2raw_frac_len = frac_len;
  int rect2raw_world_len = rect2raw_int_len + rect2raw_frac_len;
  std::vector<double> rect2raw_delta_sample;
  for (int i = 0; i < rect2raw_sample_row_num * rect2raw_sample_col_num; i++) {
    rect2raw_delta_sample.push_back(lut_info[index]);
    index++;
  }

  std::vector<std::string> rect2raw_delta_sample_bin;

  switch (int_len) {
    case 9:
      rect2raw_delta_sample_bin.reserve(rect2raw_delta_sample.size());
      for (double i : rect2raw_delta_sample) {
        rect2raw_delta_sample_bin.push_back(dec2bin_int_eight(i));
      }
      break;
    case 11:
      rect2raw_delta_sample_bin.reserve(rect2raw_delta_sample.size());
      for (double i : rect2raw_delta_sample) {
        rect2raw_delta_sample_bin.push_back(dec2bin_int_ten(i));
      }
      break;
    default:
      printf(
          "Undefined integer bit length. Only suppport int_len = 11 && int_len "
          "= 9 right now.");
  }

  std::vector<double> rect2raw_delta_sample_x, rect2raw_delta_sample_y;

  for (auto& i : rect2raw_delta_sample_bin) {
    auto _len = i.length();
    std::string _suby = i.substr(0, rect2raw_world_len);
    std::string _subx = i.substr(rect2raw_world_len, _len);
    rect2raw_delta_sample_y.push_back(bin2dec(_suby) /
                                      pow(2, rect2raw_frac_len));
    rect2raw_delta_sample_x.push_back(bin2dec(_subx) /
                                      pow(2, rect2raw_frac_len));
  }

  for (double& i : rect2raw_delta_sample_y) {
    if (i > 0.5 * pow(2, rect2raw_int_len - 1)) {
      i = i - pow(2, rect2raw_int_len);
    }
  }

  for (double& i : rect2raw_delta_sample_x) {
    if (i > 0.5 * pow(2, rect2raw_int_len - 1)) {
      i = i - pow(2, rect2raw_int_len);
    }
  }

  for (double& i : rect2raw_delta_sample_y) {
    if (i < -0.5 * pow(2, rect2raw_int_len)) {
      i = i + pow(2, rect2raw_int_len);
    }
  }

  for (double& i : rect2raw_delta_sample_x) {
    if (i < -0.5 * pow(2, rect2raw_int_len)) {
      i = i + pow(2, rect2raw_int_len);
    }
  }

  cv::Mat rect2raw_delta_samplex((int)rect2raw_sample_col_num,
                                 (int)rect2raw_sample_row_num,
                                 CV_64F,
                                 rect2raw_delta_sample_x.data());
  cv::Mat rect2raw_delta_sampley((int)rect2raw_sample_col_num,
                                 (int)rect2raw_sample_row_num,
                                 CV_64F,
                                 rect2raw_delta_sample_y.data());

  /* need transpose */
  cv::transpose(raw2rect_delta_samplex, raw2rect_delta_samplex);
  cv::transpose(raw2rect_delta_sampley, raw2rect_delta_sampley);
  cv::transpose(rect2raw_delta_samplex, rect2raw_delta_samplex);
  cv::transpose(rect2raw_delta_sampley, rect2raw_delta_sampley);

  cv::Mat Raw2RectMapX, Raw2RectMapY, Rect2RawMapX, Rect2RawMapY;
  cv::subtract(raw2rectSample_x, raw2rect_delta_samplex, Raw2RectMapX);
  cv::subtract(raw2rectSample_y, raw2rect_delta_sampley, Raw2RectMapY);
  cv::subtract(rect2rawSample_x, rect2raw_delta_samplex, Rect2RawMapX);
  cv::subtract(rect2rawSample_y, rect2raw_delta_sampley, Rect2RawMapY);

  int nr = (int)row_num;
  int nc = (int)col_num;

  xOrig2Rect = sparse2dense(nr,
                            nc / 2,
                            Raw2RectMapX,
                            raw2rectSample_x,
                            raw2rectSample_y,
                            params) -
               1;
  yOrig2Rect = sparse2dense(nr,
                            nc / 2,
                            Raw2RectMapY,
                            raw2rectSample_x,
                            raw2rectSample_y,
                            params) -
               1;
  xRect2Orig = sparse2dense(nr,
                            nc * 1,
                            Rect2RawMapX,
                            rect2rawSample_x,
                            rect2rawSample_y,
                            params) -
               1;
  yRect2Orig = sparse2dense(nr,
                            nc * 1,
                            Rect2RawMapY,
                            rect2rawSample_x,
                            rect2rawSample_y,
                            params) -
               1;
}

cv::Mat sparse2dense(int row,
                     int col,
                     cv::Mat sparseMat,
                     cv::Mat sampleX,
                     cv::Mat sampleY,
                     std::shared_ptr<useLutInitParams> params) {
  const int int_len = params->int_len, frac_len = params->frac_len;
  cv::Mat xAll, yAll;
  meshgrid(cv::Range(1, col), cv::Range(1, row), xAll, yAll);
  cv::Mat XY;
  // need to transpose, c++ is manipulating by rows whereas matlab by cols.
  xAll = xAll.t();
  yAll = yAll.t();
  cv::hconcat(xAll.reshape(0, row * col), yAll.reshape(0, row * col), XY);

  double a0, a1, a2, a3;
  double b0, b1, b2, b3;
  double c0, c1, c2, c3, c4;
  const int xrows = sampleX.rows;
  const int xcols = sampleX.cols;
  const int xyrows = XY.rows;
  cv::Mat denseMat(row * col, 1, CV_64F);

  for (int i = 0; i < xrows - 1; i++) {
    double* ptr_y = sampleY.ptr<double>(i);
    double* ptr_ynext = sampleY.ptr<double>(i + 1);
    double dltY = *ptr_ynext - *ptr_y;
    double* ptr_x = sampleX.ptr<double>(0);
    double* ptr_sparse = sparseMat.ptr<double>(i);
    double* ptr_sparse_next = sparseMat.ptr<double>(i + 1);
    for (int j = 0; j < xcols - 1; j++) {
      double mark_x = ptr_x[j];
      double mark_xnext = ptr_x[j + 1];
      double dltX = mark_xnext - mark_x;
      double dltXY = dltX * dltY;
      for (int k = 0; k < xyrows; k++) {
        int* ptr_xy = XY.ptr<int>(k);
        double* ptr_dense = denseMat.ptr<double>(k);
        if (ptr_xy[0] >= mark_x && ptr_xy[0] < mark_xnext &&
            ptr_xy[1] >= *ptr_y && ptr_xy[1] < *ptr_ynext) {
          /* handle with sparseMat */
          a0 = num2fix(ptr_sparse[j] / dltXY, int_len + frac_len - 2);
          a1 = num2fix(ptr_sparse[j + 1] / dltXY, int_len + frac_len - 2);
          a2 = num2fix(ptr_sparse_next[j] / dltXY, int_len + frac_len - 2);
          a3 = num2fix(ptr_sparse_next[j + 1] / dltXY, int_len + frac_len - 2);

          /* handle with sample */
          b0 = ((mark_xnext - ptr_xy[0]) * (*ptr_ynext - ptr_xy[1]));
          b1 = ((-mark_x + ptr_xy[0]) * (*ptr_ynext - ptr_xy[1]));
          b2 = ((mark_xnext - ptr_xy[0]) * (-*ptr_y + ptr_xy[1]));
          b3 = ((-mark_x + ptr_xy[0]) * (-*ptr_y + ptr_xy[1]));

          /* handle with denseMat and apply bilinear interpolation */
          c0 = num2fix(a0 * b0, int_len + frac_len - 2);
          c1 = num2fix(a1 * b1, int_len + frac_len - 2);
          c2 = num2fix(a2 * b2, int_len + frac_len - 2);
          c3 = num2fix(a3 * b3, int_len + frac_len - 2);
          c4 = num2fix((c0 + c1 + c2 + c3), int_len);
          ptr_dense[0] = c4;
        }
      }
    }
  }
  denseMat = denseMat.t();
  denseMat = denseMat.reshape(0, col).t();
  return denseMat;
}

// convert float number to fixed point number then return as double
double num2fix(double num, int frac_len) {
  int32_t resf;
  double resd;
  resf = double2fixed(num, frac_len);
  resd = fixed2double(resf, frac_len);
  return resd;
}

double num2fix_unsigned(double num, int frac_len) {
  int32_t resf;
  double resd, res;
  resf = double2fixed(num, frac_len);
  resd = fixed2double(resf, frac_len);
  res = (resd > 0) ? resd : 0;
  return res;
}
