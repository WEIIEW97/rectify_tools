/**
 * @file generate_lut.cc
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "generate_lut.h"

#include "gen_sample_point.h"

std::vector<unsigned int> Make_BLendian_Data(std::vector<unsigned int> number) {
  int i;
  unsigned int num[4];
  for (i = 0; i < number.size(); i++) {
    num[0] = number[i] & 0xff;
    num[1] = (number[i] >> 8) & 0xff;
    num[2] = (number[i] >> 16) & 0xff;
    num[3] = (number[i] >> 24) & 0xff;
    number[i] = (num[0] << 24) | (num[1] << 16) | (num[2] << 8) | num[3];
  }
  return number;
}

#ifndef _WIN32
/**
 * C++ version 0.4 char* style "itoa":
 * Written by Lukás Chmela
 * Released under GPLv3.
 */
static char* _itoa_s(int value, char* result, int base) {
  // check that the base if valid
  if (base < 2 || base > 36) {
    *result = '\0';
    return result;
  }

  char *ptr = result, *ptr1 = result, tmp_char;
  int tmp_value;

  do {
    tmp_value = value;
    value /= base;
    *ptr++ =
        "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxy"
        "z"[35 + (tmp_value - value * base)];
  } while (value);

  // Apply negative sign
  if (tmp_value < 0) *ptr++ = '-';
  *ptr-- = '\0';
  while (ptr1 < ptr) {
    tmp_char = *ptr;
    *ptr-- = *ptr1;
    *ptr1++ = tmp_char;
  }
  return result;
}
#endif

std::string dec2bin(double in_num, int length_num) {
  int i;
  char str_num[64];
  std::string transfor_num;

  _itoa_s(in_num, str_num, 2);
  if (length_num > strlen(str_num)) {
    for (i = 0; i < length_num - strlen(str_num); i++) {
      transfor_num += '0';
    }
  }
  transfor_num += str_num;
  return transfor_num;
}

unsigned int cal_bin2dec(std::string tr) {
  int num, com_result;
  unsigned int sum = 0;
  char a = '1';
  for (num = 0; num < tr.size(); num++) {
    if (a == tr[num]) {
      sum = sum * 2 + 1;
    } else {
      sum = sum * 2;
    }
  }
  return sum;
}

cv::Mat apply_distortion(cv::Mat x, cv::Mat k) {
  double column = x.cols;
  double col_num;
  cv::Mat r2(1, column, CV_64FC1), r4(1, column, CV_64FC1),
      r6(1, column, CV_64FC1);
  cv::Mat cdist(1, column, CV_64FC1), xd1(2, column, CV_64FC1);
  cv::Mat a1(1, column, CV_64FC1), a2(1, column, CV_64FC1),
      a3(1, column, CV_64FC1);
  cv::Mat delta_x(2, column, CV_64FC1), xd(2, column, CV_64FC1);

  for (col_num = 0; col_num < column; col_num++) {
    r2.at<double>(0, col_num) =
        pow(x.at<double>(0, col_num), 2) + pow(x.at<double>(1, col_num), 2);
    r4.at<double>(0, col_num) = pow(r2.at<double>(0, col_num), 2);
    r6.at<double>(0, col_num) = pow(r2.at<double>(0, col_num), 3);
  }

  for (col_num = 0; col_num < column; col_num++) {
    cdist.at<double>(0, col_num) =
        1 + k.at<double>(0, 0) * r2.at<double>(0, col_num) +
        k.at<double>(1, 0) * r4.at<double>(0, col_num) +
        k.at<double>(4, 0) * r6.at<double>(0, col_num);
    xd1.at<double>(0, col_num) =
        x.at<double>(0, col_num) * cdist.at<double>(0, col_num);
    xd1.at<double>(1, col_num) =
        x.at<double>(1, col_num) * cdist.at<double>(0, col_num);
  }

  for (col_num = 0; col_num < column; col_num++) {
    a1.at<double>(0, col_num) =
        2 * x.at<double>(0, col_num) * x.at<double>(1, col_num);
    a2.at<double>(0, col_num) =
        r2.at<double>(0, col_num) + 2 * pow(x.at<double>(0, col_num), 2);
    a3.at<double>(0, col_num) =
        r2.at<double>(0, col_num) + 2 * pow(x.at<double>(1, col_num), 2);
  }

  for (col_num = 0; col_num < column; col_num++) {
    delta_x.at<double>(0, col_num) =
        k.at<double>(2, 0) * a1.at<double>(0, col_num) +
        k.at<double>(3, 0) * a2.at<double>(0, col_num);
    delta_x.at<double>(1, col_num) =
        k.at<double>(2, 0) * a3.at<double>(0, col_num) +
        k.at<double>(3, 0) * a1.at<double>(0, col_num);
  }

  cv::add(xd1, delta_x, xd);

  return xd;
}

cv::Mat comp_distortion_oulu(cv::Mat xd, cv::Mat k) {
  int kk;
  double k1, k2, k3, p1, p2;
  double row = xd.rows, col = xd.cols;
  cv::Mat x(row, col, CV_64FC1);
  cv::Mat x_square(row, col, CV_64FC1);
  cv::Mat r_2(1, col, CV_64FC1);
  cv::Mat k_radial(1, col, CV_64FC1);
  cv::Mat k_radial_vconcat(row, col, CV_64FC1);
  cv::Mat r_2_square(1, col, CV_64FC1);
  cv::Mat r_2_cube(1, col, CV_64FC1);
  cv::Mat delta_x(row, col, CV_64FC1);
  cv::Mat x_0_square(1, col, CV_64FC1);
  cv::Mat x_1_square(1, col, CV_64FC1);

  k1 = k.at<double>(0, 0);
  k2 = k.at<double>(1, 0);
  k3 = k.at<double>(4, 0);
  p1 = k.at<double>(2, 0);
  p2 = k.at<double>(3, 0);

  x = xd.clone();

  for (kk = 0; kk < 20; kk++) {
    cv::pow(x, 2, x_square);
    cv::reduce(x_square, r_2, 0, cv::REDUCE_SUM);
    cv::pow(r_2, 2, r_2_square);
    cv::pow(r_2, 3, r_2_cube);
    k_radial = 1 + k1 * r_2 + k2 * r_2_square + k3 * r_2_cube;

    cv::pow(x.rowRange(0, 1), 2, x_0_square);
    cv::pow(x.rowRange(1, 2), 2, x_1_square);
    cv::vconcat(2 * p1 * ((x.rowRange(0, 1)).mul(x.rowRange(1, 2))) +
                    p2 * (r_2 + 2 * x_0_square),
                2 * p2 * ((x.rowRange(0, 1)).mul(x.rowRange(1, 2))) +
                    p1 * (r_2 + 2 * x_1_square),
                delta_x);
    cv::vconcat(k_radial, k_radial, k_radial_vconcat);
    x = (xd - delta_x) / k_radial_vconcat;
  }
  return x;
}

cv::Mat normalize_pixel(
    cv::Mat x_kk, cv::Mat fc, cv::Mat cc, cv::Mat kc, double alpha_c) {
  double row = x_kk.rows, col = x_kk.cols;
  cv::Mat xn(row, col, CV_64FC1), x_distort(row, col, CV_64FC1);

  cv::vconcat((x_kk.rowRange(0, 1) - cc.at<double>(0, 0)) / fc.at<double>(0, 0),
              (x_kk.rowRange(1, 2) - cc.at<double>(1, 0)) / fc.at<double>(1, 0),
              x_distort);
  x_distort.rowRange(0, 1) =
      x_distort.rowRange(0, 1) - alpha_c * x_distort.rowRange(1, 2);
  xn = comp_distortion_oulu(x_distort, kc);
  return xn;
}

cv::Mat Orig2Rect(cv::Mat pix,
                  cv::Mat intrMatOld,
                  cv::Mat intrMatNew,
                  cv::Mat R,
                  cv::Mat kc) {
  cv::Mat pixUndist, pixUndistR, pixRect;
  cv::Mat pix_transpose, pixUndistHomo;

  cv::transpose(pix, pix_transpose);
  cv::Mat input1 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 0),
                    intrMatOld.at<double>(1, 1));
  cv::Mat input2 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 2),
                    intrMatOld.at<double>(1, 2));
  pixUndist = normalize_pixel(pix_transpose, input1, input2, kc, 0);  //

  cv::Mat monesmat = cv::Mat::ones(1, pixUndist.cols, CV_64FC1);
  cv::vconcat(pixUndist, monesmat, pixUndistHomo);
  pixUndistR = R * pixUndistHomo;
  pixRect = intrMatNew * pixUndistR;

  cv::vconcat(pixRect.rowRange(0, 1) / pixRect.rowRange(2, 3),
              pixRect.rowRange(1, 2) / pixRect.rowRange(2, 3),
              pixRect);
  cv::transpose(pixRect, pixRect);
  return pixRect;
}

cv::Mat remapRect(cv::Mat pixRect,
                  cv::Mat KDistort,
                  cv::Mat KRect,
                  cv::Mat R,
                  cv::Mat distCoeff) {
  double alpha = 0;
  cv::Mat pixDist, pixRectHomo, rays, rays2, x, xd, px2, py2, KRectinv;
  cv::Mat monesmat = cv::Mat::ones(1, pixRect.rows, CV_64FC1);

  cv::transpose(pixRect, pixRect);
  cv::vconcat(pixRect, monesmat, pixRectHomo);
  cv::invert(KRect, KRectinv);
  rays = KRectinv * pixRectHomo;

  cv::transpose(R, R);
  rays2 = R * rays;
  cv::vconcat(rays2.rowRange(0, 1) / rays2.rowRange(2, 3),
              rays2.rowRange(1, 2) / rays2.rowRange(2, 3),
              x);

  xd = apply_distortion(x, distCoeff);

  px2 = KDistort.at<double>(0, 0) *
            (xd.rowRange(0, 1) + alpha * xd.rowRange(1, 2)) +
        KDistort.at<double>(0, 2);
  py2 =
      KDistort.at<double>(1, 1) * xd.rowRange(1, 2) + KDistort.at<double>(1, 2);

  cv::vconcat(px2, py2, pixDist);
  cv::transpose(pixDist, pixDist);

  return pixDist;
}

cv::Mat MakeOfst2(cv::Mat inValidY, cv::Mat deltaLut) {
  cv::Mat deltaLutNew, imValidYUp, imValidYDown, id;

  imValidYUp = inValidY.rowRange(0, round((float)inValidY.rows / 2));
  imValidYDown =
      inValidY.rowRange(round((float)inValidY.rows / 2), inValidY.rows);

  int ii, rows, max;
  for (ii = 0; ii < imValidYUp.cols; ii++) {
    for (rows = 0; rows < imValidYUp.rows; rows++) {
      if (imValidYUp.at<double>(rows, ii) != 0) {
        max = rows;
      }
    }
    for (rows = 0; rows < imValidYUp.rows; rows++) {
      if (imValidYUp.at<double>(rows, ii) != 0) {
        imValidYUp.at<double>(rows, ii) = imValidYUp.at<double>(max, ii);
      }
    }
  }

  for (ii = 0; ii < imValidYDown.cols; ii++) {
    for (rows = 0; rows < imValidYDown.rows; rows++) {
      if (imValidYDown.at<double>(rows, ii) != 0) {
        max = rows;
        break;
      }
    }
    for (rows = 0; rows < imValidYDown.rows; rows++) {
      if (imValidYDown.at<double>(rows, ii) != 0) {
        imValidYDown.at<double>(rows, ii) = imValidYDown.at<double>(max, ii);
      }
    }
  }

  cv::vconcat(imValidYUp, imValidYDown, inValidY);

  for (ii = 0; ii < deltaLut.cols; ii++) {
    for (rows = 0; rows < deltaLut.rows; rows++) {
      if (inValidY.at<double>(rows, ii) != 0) {
        deltaLut.at<double>(rows, ii) = inValidY.at<double>(rows, ii);
      }
    }
  }

  deltaLutNew = deltaLut;

  return deltaLutNew;
}

/**
 * @file generate_lut.cc
 *
 * @brief Generate rectified and original mapping LUT by camera intrinsics for further rectification.
 *
 * @ingroup PackageName
 * (Note: this needs exactly one @defgroup somewhere)
 *
 * @author William Wei
 * Contact: NextVPU depth division of algorithm team.
 *
 */

std::shared_ptr<genLutInitParams> initialize_(
    int w, int h, int upcrop[2], int downcrop[2], int int_len, int frac_len) {
  //   initParams* params = new initParams;
  std::shared_ptr<genLutInitParams> params(new genLutInitParams);
  params->img_size[0] = h;
  params->img_size[1] = w;
  params->upcrop[0] = upcrop[0];
  params->upcrop[1] = upcrop[1];
  params->downcrop[0] = downcrop[0];
  params->downcrop[1] = downcrop[1];
  params->int_len = int_len;
  params->frac_len = frac_len;
  return params;
}

std::shared_ptr<camParams> Get_Cam_Parameter(cv::Mat transVecT,
                                             cv::Mat intrMatOldLT,
                                             cv::Mat intrMatOldRT,
                                             cv::Mat kcLT,
                                             cv::Mat kcRT,
                                             cv::Mat intrMatNewLT,
                                             cv::Mat intrMatNewRT,
                                             cv::Mat rotMatLT,
                                             cv::Mat rotMatRT) {
  std::shared_ptr<camParams> cam_param(new camParams);
  cam_param->transVec = transVecT;
  cam_param->intrMatOldL = intrMatOldLT;
  cam_param->intrMatOldR = intrMatOldRT;
  cam_param->kcL = kcLT;
  cam_param->kcR = kcRT;
  cam_param->intrMatNewL = intrMatNewLT;
  cam_param->intrMatNewR = intrMatNewRT;
  cam_param->rotMatL = rotMatLT;
  cam_param->rotMatR = rotMatRT;
  return cam_param;
}

matGroup compute_delta(cv::Mat intrMatOld,
                       cv::Mat kc,
                       cv::Mat intrMatNew,
                       cv::Mat rotMat,
                       cv::Mat sampledX,
                       cv::Mat sampledY,
                       std::shared_ptr<genLutInitParams> params,
                       bool reverse) {
  matGroup container;
  int int_len = params->int_len, frac_len = params->frac_len;
  int total_len = int_len + frac_len;
  int nc = params->img_size[1], nr = params->img_size[0], thr;
  int lut_size[2] = {std::max(sampledY.rows, sampledY.cols),
                     std::max(sampledX.rows, sampledX.cols)};

  // repeat sample matrix
  cv::Mat xMatSampled;
  cv::repeat(sampledX, sampledY.cols, 1, xMatSampled);
  cv::Mat yMatSampled;
  cv::repeat(sampledY.t(), 1, sampledX.cols, yMatSampled);

  // make them paired
  cv::Mat pixSampledX, pixSampledY, pixSampled;
  for (int i = 0; i < xMatSampled.cols; i++) {
    pixSampledX.push_back(xMatSampled.colRange(i, i + 1));
    pixSampledY.push_back(yMatSampled.colRange(i, i + 1));
    cv::hconcat(pixSampledX, pixSampledY, pixSampled);
  }
  cv::Mat pix_rect;
  if (!reverse) {
    pix_rect = remapRect(pixSampled, intrMatOld, intrMatNew, rotMat, kc);
  } else {
    pix_rect = Orig2Rect(pixSampled, intrMatOld, intrMatNew, rotMat, kc);
  }

  cv::Mat x_orig = cv::Mat::zeros(lut_size[0], lut_size[1], CV_64F);
  cv::Mat y_orig = cv::Mat::zeros(lut_size[0], lut_size[1], CV_64F);
  for (int j = 0; j < lut_size[1]; j++) {
    for (int i = 0; i < lut_size[0]; i++) {
      double* x_orig_p = x_orig.ptr<double>(i);
      double* y_orig_p = y_orig.ptr<double>(i);
      double* pix_rect_p = pix_rect.ptr<double>(j * lut_size[0] + i);
      x_orig_p[j] = pix_rect_p[0];
      y_orig_p[j] = pix_rect_p[1];
    }
  }

  cv::Mat delta_x2, delta_y2, delta_x, delta_y;
  cv::Mat invalid_x = cv::Mat::zeros(lut_size[0], lut_size[1], CV_64F);
  cv::Mat invalid_y = cv::Mat::zeros(lut_size[0], lut_size[1], CV_64F);
  cv::Mat valid = cv::Mat::zeros(lut_size[0], lut_size[1], CV_64F);
  if (!reverse) {
    delta_x = xMatSampled - x_orig;
    delta_y = yMatSampled - y_orig;
    thr = pow(2, int_len - 2);
    // thr = 128;
  } else {
    for (int i = 0; i < x_orig.rows; i++) {
      double* x_orig_p = x_orig.ptr<double>(i);
      double* y_orig_p = y_orig.ptr<double>(i);
      double* valid_p = valid.ptr<double>(i);
      for (int j = 0; j < x_orig.cols; j++) {
        if (cvFloor(x_orig_p[j]) >= 1 && cvFloor(x_orig_p[j] <= nc)) {
          if (cvFloor(y_orig_p[j]) >= 1 && cvFloor(y_orig_p[j] <= nr)) {
            valid_p[j] = 1;
          }
        }
        if (cvCeil(x_orig_p[j]) >= 1 && cvCeil(x_orig_p[j] <= nc)) {
          if (cvCeil(y_orig_p[j]) >= 1 && cvCeil(y_orig_p[j] <= nr)) {
            valid_p[j] = 1;
          }
        }
      }
    }

    for (int j = 0; j < valid.cols; j++) {
      valid.ptr<double>(0)[j] = 0;
      valid.ptr<double>(valid.rows - 1)[j] = 0;
    }

    for (int i = 0; i < valid.rows; i++) {
      valid.ptr<double>(i)[0] = 0;
      valid.ptr<double>(i)[valid.cols - 1] = 0;
    }

    cv::Mat kernel_struct =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(valid, valid, kernel_struct);

    delta_x2 = xMatSampled - x_orig;
    delta_y2 = yMatSampled - y_orig;

    for (int i = 0; i < valid.rows; i++) {
      double* valid_p = valid.ptr<double>(i);
      double* invalid_x_p = invalid_x.ptr<double>(i);
      double* invalid_y_p = invalid_y.ptr<double>(i);
      double* delta_x_p = delta_x2.ptr<double>(i);
      double* delta_y_p = delta_y2.ptr<double>(i);
      for (int j = 0; j < valid.cols; j++) {
        if (valid_p[j] != 0) {
          invalid_x_p[j] = 0;
          invalid_y_p[j] = 0;
        } else {
          invalid_x_p[j] = delta_x_p[j];
          invalid_y_p[j] = delta_y_p[j];
        }
      }
    }

    delta_x = MakeOfst2(invalid_x, delta_x2);
    delta_y = MakeOfst2(invalid_y, delta_y2);
    thr = pow(2, int_len - 1);
    //        thr = 256;
  }
  cv::Mat delta_X, delta_Y;
  delta_X = delta_x;
  delta_Y = delta_y;
  for (int i = 0; i < delta_X.rows; i++) {
    double* x_p = delta_X.ptr<double>(i);
    double* y_p = delta_Y.ptr<double>(i);
    for (int j = 0; j < delta_X.cols; j++) {
      if (x_p[j] > thr) {
        x_p[j] = thr - 1;
      } else if (x_p[j] < -thr) {
        x_p[j] = -thr + 1;
      }

      if (y_p[j] > thr) {
        y_p[j] = thr - 1;
      } else if (y_p[j] < -thr) {
        y_p[j] = -thr + 1;
      }
    }
  }
  cv::Mat x_orig_recovered = xMatSampled - delta_X;
  cv::Mat y_orig_recovered = yMatSampled - delta_Y;

  cv::Mat dlt_x, dlt_y;
  for (int i = 0; i < x_orig_recovered.rows; i++) {
    double* x_p = x_orig_recovered.ptr<double>(i);
    for (int j = 0; j < x_orig_recovered.cols; j++) {
      x_p[j] = cvRound(pow(2, frac_len) * x_p[j]) / pow(2, frac_len);
    }
  }
  for (int i = 0; i < y_orig_recovered.rows; i++) {
    double* y_p = y_orig_recovered.ptr<double>(i);
    for (int j = 0; j < y_orig_recovered.cols; j++) {
      y_p[j] = cvRound(pow(2, frac_len) * y_p[j]) / pow(2, frac_len);
    }
  }

  dlt_x = xMatSampled - x_orig_recovered;
  dlt_y = yMatSampled - y_orig_recovered;
  container.matx = dlt_x;
  container.maty = dlt_y;

  return container;
}

template <typename Tp>
std::vector<Tp> generate_lut_1d(int total_len, const cv::Mat& src) {
  const double* val = src.ptr<double>(0);
  std::vector<Tp> res;
  Tp val_converted;
  std::string val_str;

  for (int i = 0; i < src.cols; i++) {
    if (val[i] < 0) {
      val_str = dec2bin(pow(2, total_len) + val[i] * 2, total_len);
      val_converted = cal_bin2dec(val_str);
    } else {
      val_converted = num2fix_unsigned(val[i] * 2, total_len);
    }
    res.emplace_back(val_converted);
  }
  return res;
}

template <typename Tp>
std::vector<Tp> generate_lut_2d_kernel(int total_len, const cv::Mat& src) {
  const double* val = src.ptr<double>(0);
  std::vector<Tp> res;
  for (int i = 0; i < src.rows; i++) {
    Tp val_converted;
    if (val[i] < 0) {
      std::string val_str = dec2bin(pow(2, total_len) + val[i], total_len);
      val_converted = cal_bin2dec(val_str);
    } else {
      // std::string val_str = dec2bin(val[i], total_len);
      // val_converted = cal_bin2dec(val_str);
      val_converted = num2fix_unsigned(val[i], total_len);
    }
    res.emplace_back(val_converted);
  }
  return res;
}

template <typename Tp>
std::vector<Tp> generate_lut_2d(int int_len, int frac_len, const cv::Mat& src) {
  int total_len = int_len + frac_len;
  std::vector<Tp> res;
  Tp val_converted;

  cv::Mat tmp;
  for (int i = 0; i < src.cols; i++) {
    tmp.push_back(src.colRange(i, i + 1));
  }
  tmp = tmp * pow(2, frac_len);

  res = generate_lut_2d_kernel<Tp>(total_len, tmp);
  return res;
}

template <typename Tp>
std::vector<Tp> pair_xy(std::vector<Tp> src_x,
                        std::vector<Tp> src_y,
                        int total_len) {
  std::string y_str, x_str;
  std::vector<Tp> res;
  Tp val_;
  for (int i = 0; i < src_x.size(); i++) {
    y_str = dec2bin((double)src_y[i], total_len);
    x_str = dec2bin((double)src_x[i], total_len);
    y_str += x_str;
    val_ = cal_bin2dec(y_str);
    res.emplace_back(val_);
  }
  return res;
}

template <typename Tp>
std::vector<Tp> genlut_run(std::shared_ptr<genLutInitParams> params,
                           cv::Mat intrMatOld,
                           cv::Mat kc,
                           cv::Mat intrMatNew,
                           cv::Mat rotMat) {
  int w, h;
  // if not reversing mapping, sample width is half-size.

  w = params->img_size[1];
  h = params->img_size[0];

  int int_len = params->int_len, frac_len = params->frac_len;
  cv::Mat sample_x1, sample_x2, sample_y1, sample_y2;
  genSamplePoint(w, h, sample_x1, sample_y1, sample_x2, sample_y2);

  std::vector<Tp> orig2rect_sample_x, orig2rect_sample_y, rect2orig_sample_x,
      rect2orig_sample_y;

  // for x coord should add signed postion.
  orig2rect_sample_x = generate_lut_1d<Tp>(int_len + frac_len + 1, sample_x1);
  orig2rect_sample_y = generate_lut_1d<Tp>(int_len + frac_len, sample_y1);

  rect2orig_sample_x = generate_lut_1d<Tp>(int_len + frac_len + 1, sample_x2);
  rect2orig_sample_y = generate_lut_1d<Tp>(int_len + frac_len, sample_y2);

  matGroup orig2rect_grp, rect2orig_grp;
  orig2rect_grp = compute_delta(
      intrMatOld, kc, intrMatNew, rotMat, sample_x1, sample_y1, params, true);
  rect2orig_grp = compute_delta(
      intrMatOld, kc, intrMatNew, rotMat, sample_x2, sample_y2, params, false);

  std::vector<Tp> orig2rect_vec1, orig2rect_vec2, orig2rect_vec3;
  // generate y sample
  orig2rect_vec1 = generate_lut_2d<Tp>(int_len, frac_len, orig2rect_grp.maty);
  // generate x sample
  orig2rect_vec2 = generate_lut_2d<Tp>(int_len, frac_len, orig2rect_grp.matx);
  // pair x, y to (y, x) in one line
  orig2rect_vec3 = pair_xy(orig2rect_vec2, orig2rect_vec1, int_len + frac_len);

  std::vector<Tp> rect2orig_vec1, rect2orig_vec2, rect2orig_vec3;
  rect2orig_vec1 =
      generate_lut_2d<Tp>(int_len - 1, frac_len, rect2orig_grp.maty);
  rect2orig_vec2 =
      generate_lut_2d<Tp>(int_len - 1, frac_len, rect2orig_grp.matx);
  rect2orig_vec3 =
      pair_xy(rect2orig_vec2, rect2orig_vec1, int_len + frac_len - 1);

  std::vector<Tp> lut;

  lut.reserve(HEADER_PART_LEN + orig2rect_sample_x.size() +
              orig2rect_sample_y.size() + rect2orig_sample_x.size() +
              rect2orig_sample_y.size() + orig2rect_vec3.size() +
              rect2orig_vec3.size());

  // some fuckin fixed header info
  lut.emplace_back(163);
  lut.emplace_back(params->img_size[1]);
  lut.emplace_back(params->img_size[0]);
  lut.emplace_back((Tp)sample_y2.cols);
  lut.emplace_back((Tp)sample_x2.cols);
  lut.emplace_back((Tp)sample_y1.cols);
  lut.emplace_back((Tp)sample_x1.cols);
  lut.emplace_back(params->upcrop[1]);
  lut.emplace_back(params->upcrop[0]);
  lut.emplace_back(params->downcrop[1]);
  lut.emplace_back(params->downcrop[0]);
  lut.emplace_back(HEADER_PART_END);
  lut.emplace_back((Tp)(orig2rect_sample_x.size() + orig2rect_sample_y.size() +
                        rect2orig_sample_x.size() + rect2orig_sample_y.size() +
                        orig2rect_vec3.size() + rect2orig_vec3.size()));
  lut.emplace_back(BODY_PART_BEGIN);
  EMPLACE_BACK(orig2rect_sample_x, lut);
  EMPLACE_BACK(orig2rect_sample_y, lut);
  EMPLACE_BACK(rect2orig_sample_x, lut);
  EMPLACE_BACK(rect2orig_sample_y, lut);
  EMPLACE_BACK(orig2rect_vec3, lut);
  EMPLACE_BACK(rect2orig_vec3, lut);

  return lut;
}

template <typename Tp>
void save_vector_to_txt(const char* path, std::vector<Tp> src) {
  FILE* stream;
  stream = fopen(path, "w");
  for (auto i : src) {
    fprintf(stream, "%u\n", i);
  }
  fclose(stream);
}

template <typename Tp>
stereoCam<Tp> stereo_run(std::shared_ptr<genLutInitParams> params,
                         std::shared_ptr<camParams> cam_params) {
  int height = params->img_size[0];
  int width = params->img_size[1];
  int image_size_t[2];
  int cheat;
  float scale;
  stereoCam<Tp> con;
  std::vector<Tp> left_, right_;
  if (width > 1920) {
    cheat = 1;
    scale = width / 1920;
    image_size_t[0] = 1080;
    image_size_t[1] = 1920;
  } else {
    cheat = 0;
    scale = 1.0f;
    image_size_t[0] = height;
    image_size_t[1] = width;
  }

  cv::Mat intrMatOldLN, intrMatOldRN, intrMatNewLN, intrMatNewRN;
  if (cheat) {
    intrMatOldLN = cam_params->intrMatOldL.clone();
    intrMatOldRN = cam_params->intrMatOldR.clone();
    intrMatNewLN = cam_params->intrMatNewL.clone();
    intrMatNewRN = cam_params->intrMatNewR.clone();
    for (int row = 0; row < cam_params->intrMatOldL.rows; row++) {
      double* old_l_p = intrMatOldLN.ptr<double>(row);
      double* old_r_p = intrMatOldRN.ptr<double>(row);
      double* new_l_p = intrMatNewLN.ptr<double>(row);
      double* new_r_p = intrMatNewRN.ptr<double>(row);
      for (int col = 0; col < cam_params->intrMatOldL.cols; col++) {
        old_l_p[col] = old_l_p[col] / scale;
        old_r_p[col] = old_r_p[col] / scale;
        new_l_p[col] = new_l_p[col] / scale;
        new_r_p[col] = new_r_p[col] / scale;
      }
    }
    intrMatOldLN.ptr<double>(2)[2] = 1;
    intrMatOldRN.ptr<double>(2)[2] = 1;
    intrMatNewLN.ptr<double>(2)[2] = 1;
    intrMatNewRN.ptr<double>(2)[2] = 1;
    // left cam
    left_ = genlut_run<Tp>(params,
                           intrMatOldLN,
                           cam_params->kcL,
                           intrMatNewLN,
                           cam_params->rotMatL);
    // right cam
    right_ = genlut_run<Tp>(params,
                            intrMatOldRN,
                            cam_params->kcR,
                            intrMatNewRN,
                            cam_params->rotMatR);

  } else {
    // left cam
    left_ = genlut_run<Tp>(params,
                           cam_params->intrMatOldL,
                           cam_params->kcL,
                           cam_params->intrMatNewL,
                           cam_params->rotMatL);
    // right cam
    right_ = genlut_run<Tp>(params,
                            cam_params->intrMatOldR,
                            cam_params->kcR,
                            cam_params->intrMatNewR,
                            cam_params->rotMatR);

    con.left_cam = left_;
    con.right_cam = right_;
  }
  return con;
}

template <typename Tp>
void write_txtbin(std::shared_ptr<genLutInitParams> params,
                  std::vector<Tp> tbdata,
                  char const* path,
                  char const* which_cam) {
  char tr[300];
  int i;
  FILE* stream;
  int width = params->img_size[1];
  int height = params->img_size[0];
  char w[10], h[10];
#ifdef _WIN32
  sprintf_s(w, "%d", width);
  sprintf_s(h, "%d", height);

  /**
   * @brief Writing .txt file.
   * 
   */
  strcpy_s(tr, path);
  strcat_s(tr, "LutDec");  //LutDec
  strcat_s(tr, whichCam);
  strcat_s(tr, "_");     //_
  strcat_s(tr, w);       //w
  strcat_s(tr, "_");     //_
  strcat_s(tr, h);       //h
  strcat_s(tr, ".txt");  //.txt
  fopen_s(&stream, tr, "w");
  for (auto i : tbdata) {
    fprintf(stream, "lld\n", i);
  }
  fclose(stream);

  /**
   * @brief Writing .bin file.
   * 
   */
  std::vector<Tp> bin_data;
  bin_data = Make_Blendian_Data(tbdata);
  strcpy_s(tr, path);
  strcat_s(tr, "LutBin");  //LutBin
  strcat_s(tr, whichCam);
  strcat_s(tr, "_");     //_
  strcat_s(tr, w);       //w
  strcat_s(tr, "_");     //_
  strcat_s(tr, h);       //h
  strcat_s(tr, ".bin");  //.bin
  fopen_s(&stream, tr, "wb");
  fwrite(bin_data.data(), 4, bin_data.size(), stream);
  fclose(stream);
#else
  sprintf(w, "%d", width);
  sprintf(h, "%d", height);

  /**
   * @brief Writing .txt file.
   * 
   */
  strcpy(tr, path);
  strcat(tr, "LutDec");  //LutDec
  strcat(tr, which_cam);
  strcat(tr, "_");     //_
  strcat(tr, w);       //w
  strcat(tr, "_");     //_
  strcat(tr, h);       //h
  strcat(tr, ".txt");  //.txt
  stream = fopen(tr, "w");
  for (auto i : tbdata) {
    fprintf(stream, "%lld\n", i);
  }
  fclose(stream);

  /**
   * @brief Writing .bin file.
   * 
   */
  std::vector<Tp> bin_data;
  bin_data = Make_BLendian_Data(tbdata);
  strcpy(tr, path);
  strcat(tr, "LutBin");  //LutBin
  strcat(tr, which_cam);
  strcat(tr, "_");     //_
  strcat(tr, w);       //w
  strcat(tr, "_");     //_
  strcat(tr, h);       //h
  strcat(tr, ".bin");  //.bin
  stream = fopen(tr, "wb");
  fwrite(bin_data.data(), 4, bin_data.size(), stream);
  fclose(stream);
#endif
}

template <typename Tp>
int NVP_cam_writefile(std::shared_ptr<genLutInitParams> params,
                      char const* path,
                      bool is_mono,
                      char const* name_caml,
                      char const* name_camr,
                      stereoCam<Tp> container) {
  int h = params->img_size[0];
  int w = params->img_size[1];

  if (w > 1920) return Imgsize_error;

  std::vector<Tp> cam_left;
  std::vector<Tp> cam_right;

  cam_left = container.left_cam;
  cam_right = container.right_cam;

  if (is_mono) {
    write_txtbin(params, cam_left, path, name_caml);
  } else {
    write_txtbin(params, cam_left, path, name_caml);
    write_txtbin(params, cam_left, path, name_camr);
  }

  return Nvp_Success;
}
