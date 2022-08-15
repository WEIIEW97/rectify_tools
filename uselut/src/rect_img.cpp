/**
 * @file rect_img.cpp
 * @author William Wei (wei.wei@nextvpu.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "rect_img.h"

#include "lut_parser.h"
#include "utils.h"

rectBuffer rect_img(const cv::Mat& xOrig2Rect,
                    const cv::Mat& yOrig2Rect,
                    cv::Mat xRect2Orig,
                    cv::Mat yRect2Orig,
                    const cv::Mat& image,
                    std::shared_ptr<useLutInitParams> params) {
  rectBuffer rect_buffer;

  const int nr = image.rows;
  const int nc = image.cols;

  const int int_len = params->int_len, frac_len = params->frac_len;
  cv::Mat pix_write;

  int y_size = nr;

  std::vector<double> y_sampled;
  y_sampled.reserve(y_size);
  for (int i = 0; i < y_size; i++) {
    y_sampled.emplace_back(i);
  }

  // TODO: add crop here

  /* retrieve BGR channel separately */
  cv::Mat img_r, img_g, img_b;
  cv::extractChannel(image, img_r, 2);
  cv::extractChannel(image, img_g, 1);
  cv::extractChannel(image, img_b, 0);

  for (int idx = 0; idx < (nr - 1); idx++) {
    cv::Mat pixRect_1;
    cv::Mat x_slice, y_slice;

    cv::transpose(
        xOrig2Rect(cv::Range((int)y_sampled[idx], (int)y_sampled[idx] + 1),
                   cv::Range::all()),
        x_slice);
    cv::transpose(
        yOrig2Rect(cv::Range((int)y_sampled[idx], (int)y_sampled[idx] + 1),
                   cv::Range::all()),
        y_slice);
    cv::hconcat(x_slice, y_slice, pixRect_1);

    // scale=2 was wrong in original code
    // ceiling operation for every pixRect_1 ???
    for (int i = 0; i < pixRect_1.rows; i++) {
      double* ptr = pixRect_1.ptr<double>(i);
      for (int j = 0; j < 2; j++) {  // dimension is (N, 2)
        ptr[j] = cvCeil(ptr[j]);     // this may not work sice ceil(double
                                     // num) but not ceil(int num)
      }
    }

    cv::Mat flag =
        pixRect_1.colRange(0, 1) >= 0 & pixRect_1.colRange(0, 1) < nc &
        pixRect_1.colRange(1, 2) >= 0 & pixRect_1.colRange(1, 2) < nr;
    cv::Mat mask;
    hconcat(flag, flag, mask);

    cv::Mat res;
    pixRect_1.copyTo(res, mask);

    // take elements which != 0;
    cv::Mat pixRect;
    for (int i = 0; i < flag.rows; i++) {
      uint8_t* ptr = flag.ptr<uint8_t>(i);
      if (ptr[0] != 0) {
        pixRect.push_back(res.row(i));
      }
    }

    if (!pixRect.empty()) {
      cv::Mat startPixRect =
          pixRect.rowRange(0, 1);  // [0, 1) left closed right open.

      cv::Mat pix = startPixRect;

      cv::Mat pix_found, pix_holdup1, pix_holdup2, pix_holddown1, pix_holddown2;

      for (int j = (int)startPixRect.ptr<double>(0)[0]; j < nc; j++) {
        cv::Mat pixTmp_l = cv::Mat::ones(5, 1, CV_64F) * j;  // repmat(j, 5, 1)
        cv::Mat pixTmp_r(5, 1, CV_64F);
        double center_p = pix.ptr<double>(0)[1];
        pixTmp_r.ptr<double>(0)[0] = center_p - 2;
        pixTmp_r.ptr<double>(1)[0] = center_p - 1;
        pixTmp_r.ptr<double>(2)[0] = center_p;
        pixTmp_r.ptr<double>(3)[0] = center_p + 1;
        pixTmp_r.ptr<double>(4)[0] = center_p + 2;

        cv::Mat pixTmp;
        cv::hconcat(pixTmp_l, pixTmp_r, pixTmp);

        cv::Mat flagIn(5, 1, CV_64F);
        flagIn = pixTmp.colRange(0, 1) >= 0 & pixTmp.colRange(0, 1) < nc &
                 pixTmp.colRange(1, 2) >= 0 & pixTmp.colRange(1, 2) < nr;

        cv::Mat mask_height, mask_width;
        pixTmp.colRange(1, 2).copyTo(mask_height, flagIn);
        pixTmp.colRange(0, 1).copyTo(mask_width, flagIn);

        // convert non-zero element to vector for mask_height and
        // mask_width
        cv::Mat pixTmp_height, pixTmp_width, index;

        for (int i = 0; i < flagIn.rows; i++) {
          uint8_t* ptr = flagIn.ptr<uint8_t>(i);
          if (ptr[0] != 0) {
            pixTmp_height.push_back(mask_height.row(i));
            pixTmp_width.push_back(mask_width.row(i));
          }
        }

        for (int i = 0; i < pixTmp_width.rows; i++) {
          index.push_back(sub2ind_along_y(nr,
                                          nc,
                                          (int)pixTmp_height.at<double>(i),
                                          (int)pixTmp_width.at<double>(i)));
        }

        cv::Mat bufferTmp_x, bufferTmp_y, rect2orig_tmp_in;
        for (int i = 0; i < index.rows; i++) {
          bufferTmp_x.push_back(xRect2Orig.at<double>(index.at<int>(i)));
          bufferTmp_y.push_back(yRect2Orig.at<double>(index.at<int>(i)));
        }
        cv::hconcat(bufferTmp_x, bufferTmp_y, rect2orig_tmp_in);
        cv::Mat rect2orig_tmp =
            cv::Mat::ones(pixTmp.rows, pixTmp.cols, CV_64F) * -10;

        std::vector<int> flag_index;
        for (int i = 0; i < flagIn.rows; i++) {
          uint8_t* ptr = flagIn.ptr<uint8_t>(i);
          if (ptr[0] != 0) {
            flag_index.push_back(i);
          }
        }

        for (int row = 0; row < rect2orig_tmp.rows; row++) {
          double* ptr = rect2orig_tmp.ptr<double>(row);
          for (int by = 0; by < flag_index.size(); by++) {
            double* ptr_in = rect2orig_tmp_in.ptr<double>(by);
            if (row == flag_index[by]) {
              ptr[0] = ptr_in[0];
              ptr[1] = ptr_in[1];
            }
          }
        }

        cv::Mat _rect2orig_tmp(rect2orig_tmp.rows, rect2orig_tmp.cols, CV_64F);
        for (int row = 0; row < rect2orig_tmp.rows; row++) {
          double* _ptr = _rect2orig_tmp.ptr<double>(row);
          double* ptr = rect2orig_tmp.ptr<double>(row);
          for (int col = 0; col < rect2orig_tmp.cols; col++) {
            _ptr[col] = num2fix_unsigned(ptr[col], int_len);
          }
        }

        std::vector<double> flag_tmp;
        for (int i = 0; i < _rect2orig_tmp.rows; i++) {
          double* _ptr = _rect2orig_tmp.ptr<double>(i);
          double* ptr = rect2orig_tmp.ptr<double>(i);
          double* tmp = pixTmp.ptr<double>(i);
          if (cvFloor(_ptr[0]) >= 0 && cvFloor(_ptr[0]) <= (nc - 2) &&
              ptr[1] >= y_sampled[idx] && ptr[1] < y_sampled[idx + 1] &&
              ptr[1] >= 0 && ptr[1] < nr && tmp[1] >= 0 && tmp[1] < nr) {
            flag_tmp.push_back(i);
          }
        }

        if (!flag_tmp.empty()) {
          // matlab version is: `ismember(2+1, flag_tmp)`, guessing
          // because the index difference between c++ and matlab
          if (ismember<double>(2, flag_tmp)) {
            pix_found.push_back(pixTmp.rowRange(2, 3));
          }
          if (ismember<double>(2, flag_tmp) &&
              ((!ismember<double>(1, flag_tmp) && !pix_holdup1.empty()) ||
               (!ismember<double>(0, flag_tmp) && !pix_holdup2.empty()))) {
            pix_write.push_back(pix_holdup1);
            pix_write.push_back(pix_holdup2);
            pix_holdup1.release();
            pix_holdup2.release();
          }
          if (ismember<double>(1, flag_tmp) || ismember<double>(0, flag_tmp)) {
            if (ismember<double>(1, flag_tmp)) {
              pix_holdup1.push_back(pixTmp.rowRange(1, 2));
            }
            if (ismember<double>(0, flag_tmp)) {
              pix_holdup2.push_back(pixTmp.rowRange(0, 1));
            }
            if (!ismember<double>(2, flag_tmp)) {
              pix_write.push_back(pix_found);
              pix_found = pix_holdup1;
              pix_holdup1 = pix_holdup2;
              pix_holdup2.release();
              pix = pixTmp.rowRange(1, 2);
            }
          }
          if (ismember<double>(2, flag_tmp) &&
              ((!ismember<double>(3, flag_tmp) && !pix_holddown1.empty()) ||
               (!ismember<double>(4, flag_tmp) && !pix_holddown2.empty()))) {
            pix_write.push_back(pix_holddown1);
            pix_write.push_back(pix_holddown2);
            pix_holddown1.release();
            pix_holddown2.release();
          }
          if (ismember<double>(3, flag_tmp) || ismember<double>(4, flag_tmp)) {
            if (ismember<double>(3, flag_tmp)) {
              pix_holddown1.push_back(pixTmp.rowRange(3, 4));
            }
            if (ismember<double>(4, flag_tmp)) {
              pix_holddown2.push_back(pixTmp.rowRange(4, 5));
            }
            if (!ismember<double>(2, flag_tmp)) {
              pix_write.push_back(pix_found);
              pix_found = pix_holddown1;
              pix_holddown1 = pix_holddown2;
              pix_holddown2.release();
              pix = pixTmp.rowRange(3, 4);
            }
          }
        } else
          continue;
      }
      // pix_left means pixels which are not being handled yet. Not the
      // left-hand side left.
      cv::Mat pix_left;
      pix_left.push_back(pix_holdup1);
      pix_left.push_back(pix_holdup2);
      pix_left.push_back(pix_holddown1);
      pix_left.push_back(pix_holddown2);

      pix_write.push_back(pix_found);
      if (!pix_left.empty()) {
        pix_write.push_back(pix_left);
      }
    } else
      continue;
  }

  std::vector<int> prop_idx;
  for (int i = 0; i < pix_write.rows; i++) {
    double* ptr = pix_write.ptr<double>(i);
    if (ptr[0] >= 0 && ptr[0] < nc && ptr[1] >= 0 && ptr[1] < nr) {
      prop_idx.push_back(sub2ind_along_y(nr, nc, (int)ptr[1], (int)ptr[0]));
    }
  }

  rect_buffer.rect_idx = prop_idx;

  cv::Mat x_orig, y_orig, xy_orig_tmp;
  for (int i : prop_idx) {
    x_orig.push_back(xRect2Orig.at<double>(i));
    y_orig.push_back(yRect2Orig.at<double>(i));
  }
  cv::hconcat(x_orig, y_orig, xy_orig_tmp);

  cv::Mat xy_orig;
  xy_orig = cv::Mat::zeros(xy_orig_tmp.rows, xy_orig_tmp.cols, CV_64F);
  for (int row = 0; row < xy_orig_tmp.rows; row++) {
    double* ptr = xy_orig.ptr<double>(row);
    double* ptr_tmp = xy_orig_tmp.ptr<double>(row);
    for (int col = 0; col < xy_orig_tmp.cols; col++) {
      ptr[col] = num2fix_unsigned(ptr_tmp[col], int_len);
    }
  }

  // floor elements in xy_orig
  cv::Mat xy_orig_int;
  xy_orig_int = cv::Mat::zeros(xy_orig.rows, xy_orig.cols, CV_64F);
  for (int row = 0; row < xy_orig.rows; row++) {
    double* ptr_int = xy_orig_int.ptr<double>(row);
    double* ptr = xy_orig.ptr<double>(row);
    for (int col = 0; col < xy_orig.cols; col++) {
      ptr_int[col] = cvFloor(ptr[col]);
    }
  }
  cv::Mat xy_orig_frac(xy_orig.rows, xy_orig.cols, CV_64F);
  cv::subtract(xy_orig, xy_orig_int, xy_orig_frac);

  // cv::Mat final_flag, final_mask, final_xy_orig_int, final_xy_orig_frac;
  // final_flag = xy_orig_int.colRange(0, 1) >= 0 &
  //              xy_orig_int.colRange(0, 1) < (nc - 1) &
  //              xy_orig_int.colRange(1, 2) >= 0 &
  //              xy_orig_int.colRange(1, 2) < (nr - 1);
  // cv::hconcat(final_flag, final_flag, final_mask);
  // xy_orig_int.copyTo(final_xy_orig_int, final_mask);
  // xy_orig_frac.copyTo(final_xy_orig_frac, final_mask);

  // cv::Mat final_idx, prop_idx_mat;
  // // convert vector prop_idx to cv::mat prop_idx_mat
  // prop_idx_mat = cv::Mat(prop_idx);
  // prop_idx_mat.copyTo(final_idx, final_flag);

  // std::vector<std::tuple<int, int>> coordinates;
  // coordinates.reserve(final_idx.rows);
  // for (int i = 0; i < final_idx.rows; i++) {
  //     // use ind2sub here, not ind2sub_along_y
  //     coordinates.emplace_back(ind2sub(nr, nc, final_idx.at<int>(i)));
  // }

  // std::vector<int> ind;
  // ind.reserve(coordinates.size());
  // // for (auto [x, y] : coordinates) {
  // //     ind.push_back(sub2ind_along_y(nr / scale, nc / scale, y, x));
  // // }
  // for (auto& iter : coordinates) {
  //     ind.push_back(sub2ind_along_y(nr / scale, nc / scale,
  //     std::get<1>(iter),
  //                                   std::get<0>(iter)));
  // }

  cv::Mat img_rect;
  img_rect = bilinear_remap(
      img_r, img_g, img_b, xy_orig_int, xy_orig_frac, prop_idx, params);

  rect_buffer.rect_img = img_rect;
  return rect_buffer;
}

cv::Mat bilinear_remap(const cv::Mat& img_r,
                       const cv::Mat& img_g,
                       const cv::Mat& img_b,
                       const cv::Mat& final_xy_orig_int,
                       const cv::Mat& final_xy_orig_frac,
                       std::vector<int> ind,
                       std::shared_ptr<useLutInitParams> params) {
  const int nr = img_r.rows;
  const int nc = img_r.cols;

  const int int_len = params->int_len, frac_len = params->frac_len;

  cv::Mat bilinear_pix_floor1, bilinear_pix_floor2, bilinear_pix_floor3,
      bilinear_pix_floor4;

  bilinear_pix_floor1 = final_xy_orig_int;
  cv::hconcat(final_xy_orig_int.colRange(0, 1) + 1,
              final_xy_orig_int.colRange(1, 2),
              bilinear_pix_floor2);
  cv::hconcat(final_xy_orig_int.colRange(0, 1),
              final_xy_orig_int.colRange(1, 2) + 1,
              bilinear_pix_floor3);
  cv::hconcat(final_xy_orig_int.colRange(0, 1) + 1,
              final_xy_orig_int.colRange(1, 2) + 1,
              bilinear_pix_floor4);

  std::vector<int> bilinear_ind1, bilinear_ind2, bilinear_ind3, bilinear_ind4;
  for (int i = 0; i < final_xy_orig_int.rows; i++) {
    double* ptr1 = bilinear_pix_floor1.ptr<double>(i);
    double* ptr2 = bilinear_pix_floor2.ptr<double>(i);
    double* ptr3 = bilinear_pix_floor3.ptr<double>(i);
    double* ptr4 = bilinear_pix_floor4.ptr<double>(i);

    bilinear_ind1.push_back(
        sub2ind_along_y(nr, nc, (int)ptr1[1], (int)ptr1[0]));

    bilinear_ind2.push_back(
        sub2ind_along_y(nr, nc, (int)ptr2[1], (int)ptr2[0]));

    bilinear_ind3.push_back(
        sub2ind_along_y(nr, nc, (int)ptr3[1], (int)ptr3[0]));

    bilinear_ind4.push_back(
        sub2ind_along_y(nr, nc, (int)ptr4[1], (int)ptr4[0]));
  }

  cv::Mat coeff1 = (1 - final_xy_orig_frac.colRange(1, 2))
                       .mul((1 - final_xy_orig_frac.colRange(0, 1)));
  cv::Mat coeff2 = (1 - final_xy_orig_frac.colRange(1, 2))
                       .mul(final_xy_orig_frac.colRange(0, 1));
  cv::Mat coeff3 = (final_xy_orig_frac.colRange(1, 2))
                       .mul((1 - final_xy_orig_frac.colRange(0, 1)));
  cv::Mat coeff4 = (final_xy_orig_frac.colRange(1, 2))
                       .mul(final_xy_orig_frac.colRange(0, 1));

  for (int i = 0; i < coeff1.rows; i++) {
    coeff1.ptr<double>(i)[0] =
        num2fix_unsigned(coeff1.ptr<double>(i)[0], int_len);
    coeff2.ptr<double>(i)[0] =
        num2fix_unsigned(coeff2.ptr<double>(i)[0], int_len);
    coeff3.ptr<double>(i)[0] =
        num2fix_unsigned(coeff3.ptr<double>(i)[0], int_len);
    coeff4.ptr<double>(i)[0] =
        num2fix_unsigned(coeff4.ptr<double>(i)[0], int_len);
  }

  // for (int i = 0; i < coeff2.rows; i++) {
  //     coeff2.at<double>(i, 0) = num2fix_unsigned(coeff2.at<double>(i, 0),
  //     9);
  // }
  // for (int i = 0; i < coeff3.rows; i++) {
  //     coeff3.at<double>(i, 0) = num2fix_unsigned(coeff3.at<double>(i, 0),
  //     9);
  // }
  // for (int i = 0; i < coeff4.rows; i++) {
  //     coeff4.at<double>(i, 0) = num2fix_unsigned(coeff4.at<double>(i, 0),
  //     9);
  // }

  std::vector<double> coord1_r, coord2_r, coord3_r, coord4_r;

  coordinate_generator(coord1_r, coeff1, img_r, bilinear_ind1, int_len);
  coordinate_generator(coord2_r, coeff2, img_r, bilinear_ind2, int_len);
  coordinate_generator(coord3_r, coeff3, img_r, bilinear_ind3, int_len);
  coordinate_generator(coord4_r, coeff4, img_r, bilinear_ind4, int_len);

  std::vector<double> coord1_g, coord2_g, coord3_g, coord4_g;

  coordinate_generator(coord1_g, coeff1, img_g, bilinear_ind1, int_len);
  coordinate_generator(coord2_g, coeff2, img_g, bilinear_ind2, int_len);
  coordinate_generator(coord3_g, coeff3, img_g, bilinear_ind3, int_len);
  coordinate_generator(coord4_g, coeff4, img_g, bilinear_ind4, int_len);

  std::vector<double> coord1_b, coord2_b, coord3_b, coord4_b;

  coordinate_generator(coord1_b, coeff1, img_b, bilinear_ind1, int_len);
  coordinate_generator(coord2_b, coeff2, img_b, bilinear_ind2, int_len);
  coordinate_generator(coord3_b, coeff3, img_b, bilinear_ind3, int_len);
  coordinate_generator(coord4_b, coeff4, img_b, bilinear_ind4, int_len);

  cv::Mat r_rect, g_rect, b_rect;
  r_rect = cv::Mat::zeros(nr, nc, CV_8UC1);
  g_rect = cv::Mat::zeros(nr, nc, CV_8UC1);
  b_rect = cv::Mat::zeros(nr, nc, CV_8UC1);

  for (int i = 0; i < ind.size(); i++) {
    b_rect.at<uint8_t>(ind[i]) =
        floor(coord1_b[i] + coord2_b[i] + coord3_b[i] + coord4_b[i]);
    g_rect.at<uint8_t>(ind[i]) =
        floor(coord1_g[i] + coord2_g[i] + coord3_g[i] + coord4_g[i]);
    r_rect.at<uint8_t>(ind[i]) =
        floor(coord1_r[i] + coord2_r[i] + coord3_r[i] + coord4_r[i]);
  }

  cv::Mat _buffer[3] = {b_rect, g_rect, r_rect};
  cv::Mat img_rect;
  cv::merge(_buffer, 3, img_rect);

  return img_rect;
}

void coordinate_generator(std::vector<double>& vec,
                          cv::Mat coeff,
                          cv::Mat img_ch,
                          std::vector<int> index,
                          int frac_len) {
  for (int i = 0; i < coeff.rows; i++) {
    vec.emplace_back(num2fix_unsigned(
        coeff.ptr<double>(i)[0] * double(img_ch.at<uint8_t>(index[i])),
        frac_len));
  }
}
