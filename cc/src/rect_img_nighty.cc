#include "lut_parser.h"
#include "rect_img.h"
#include "utils.h"

rectBuffer rect_img(const cv::Mat& xOrig2Rect, const cv::Mat& yOrig2Rect,
                    cv::Mat xRect2Orig, cv::Mat yRect2Orig,
                    const cv::Mat& image, int scale) {
  rectBuffer rect_buffer;

  const int orig_nr = image.rows;
  const int orig_nc = image.cols;

  const int nr = scale * orig_nr;
  const int nc = scale * orig_nc;

  cv::Mat pix_write;

  int y_size = (nr - scale) / scale + 1;

  std::vector<double> y_sampled;
  y_sampled.reserve(y_size + 1);
  for (int i = 0; i < y_size; i++) {
    y_sampled.emplace_back(scale + scale * i - 1);
  }

  if (y_sampled.back() != nr - 1) {
    y_sampled.push_back(nr - 1);
  }

  // TODO: add crop here

  /* retrieve BGR channel separately */
  cv::Mat img_r, img_g, img_b;
  cv::extractChannel(image, img_r, 2);
  cv::extractChannel(image, img_g, 1);
  cv::extractChannel(image, img_b, 0);

  for (int ii = scale - 1; ii < scale * (nr - 1); ii += scale) {
    int idx = ii / scale;

    cv::Mat pixRect_1;
    cv::Mat x_slice, y_slice;

    cv::transpose(
        xOrig2Rect(cv::Range((int)y_sampled[idx], (int)y_sampled[idx] + scale),
                   cv::Range::all()),
        x_slice);
    cv::transpose(
        yOrig2Rect(cv::Range((int)y_sampled[idx], (int)y_sampled[idx] + scale),
                   cv::Range::all()),
        y_slice);
    cv::hconcat(x_slice, y_slice, pixRect_1);

    // scale=2 was wrong in original code
    // ceiling operation for every pixRect_1 ???
    for (int i = 0; i < pixRect_1.size[0]; i++) {
      auto* ptr = pixRect_1.ptr<double>(i);
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
      if (flag.ptr<uint8_t>(i)[0] != 0) {
        pixRect.push_back(res.row(i));
      }
    }

    // ??? why would we do this?
    if (!pixRect.empty()) {
      auto* ptr = pixRect.ptr<double>(0);
      if (int(ptr[0]) % scale != 0) {
        ptr[0] = ptr[0] - 1;
      } else if (int(ptr[1]) % scale != 0) {
        ptr[1] = ptr[1] - 1;
      }
    } else
      continue;

    if (!pixRect.empty()) {
      cv::Mat startPixRect =
          pixRect.rowRange(0, 1).clone();  // [0, 1) left closed right open.

      cv::Mat pix = startPixRect;

      cv::Mat pix_found, pix_holdup1, pix_holdup2, pix_holddown1, pix_holddown2;

      for (int j = (int)startPixRect.ptr<double>(0)[0]; j < nc; j++) {
        cv::Mat pixTmp_l = cv::Mat::ones(5, 1, CV_64F) * j;  // repmat(j, 5, 1)
        cv::Mat pixTmp_r(5, 1, CV_64F);
        auto* tmpr_ptr = pixTmp_r.ptr<double>(0);
        auto* pix_ptr = pix.ptr<double>(0);
        tmpr_ptr[0] = pix_ptr[1] - 2 * scale;
        tmpr_ptr[1] = pix_ptr[1] - 1 * scale;
        tmpr_ptr[2] = pix_ptr[1];
        tmpr_ptr[3] = pix_ptr[1] + 1 * scale;
        tmpr_ptr[4] = pix_ptr[1] + 2 * scale;

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
          if (flagIn.ptr<uint8_t>(i)[0] != 0) {
            pixTmp_height.push_back(mask_height.row(i));
            pixTmp_width.push_back(mask_width.row(i));
          }
        }

        for (int i = 0; i < pixTmp_width.rows; i++) {
          index.push_back(sub2ind_along_y(nr, nc,
                                          (int)pixTmp_height.ptr<double>(i)[0],
                                          (int)pixTmp_width.ptr<double>(i)[0]));
        }

        cv::Mat bufferTmp_x, bufferTmp_y, rect2orig_tmp_in;
        for (int i = 0; i < index.rows; i++) {
          bufferTmp_x.push_back(xRect2Orig.at<double>(*index.ptr<int>(i)));
          bufferTmp_y.push_back(yRect2Orig.at<double>(*index.ptr<int>(i)));
        }
        cv::hconcat(bufferTmp_x, bufferTmp_y, rect2orig_tmp_in);
        cv::Mat rect2orig_tmp =
            cv::Mat::ones(pixTmp.rows, pixTmp.cols, CV_64F) * -10;

        std::vector<int> flag_index;
        for (int i = 0; i < flagIn.rows; i++) {
          if (flagIn.ptr<uint8_t>(i)[0] != 0) {
            flag_index.push_back(i);
          }
        }

        for (int row = 0; row < rect2orig_tmp.rows; row++) {
          auto* tmp_ptr = rect2orig_tmp.ptr<double>(row);
          for (int by = 0; by < flag_index.size(); by++) {
            if (row == flag_index[by]) {
              auto* tmp_in_ptr = rect2orig_tmp_in.ptr<double>(by);
              tmp_ptr[0] = tmp_in_ptr[0];
              tmp_ptr[1] = tmp_in_ptr[1];
            }
          }
        }

        cv::Mat _rect2orig_tmp(rect2orig_tmp.rows, rect2orig_tmp.cols, CV_64F);
        for (int row = 0; row < rect2orig_tmp.rows; row++) {
          for (int col = 0; col < rect2orig_tmp.cols; col++) {
            _rect2orig_tmp.ptr<double>(row)[col] = num2fix_unsigned(
                rect2orig_tmp.ptr<double>(row)[col] / scale, 9);
          }
        }

        std::vector<double> flag_tmp;
        for (int i = 0; i < _rect2orig_tmp.rows; i++) {
          if (cvFloor(_rect2orig_tmp.ptr<double>(i)[0]) >= 0 &&
              cvFloor(_rect2orig_tmp.ptr<double>(i)[0]) <= (nc / scale - 2) &&
              rect2orig_tmp.ptr<double>(i)[1] >= y_sampled[idx] &&
              rect2orig_tmp.ptr<double>(i)[1] < y_sampled[idx + 1] &&
              rect2orig_tmp.ptr<double>(i)[1] >= 0 &&
              rect2orig_tmp.ptr<double>(i)[1] < nr &&
              pixTmp.ptr<double>(i)[1] >= 0 && pixTmp.ptr<double>(i)[1] < nr) {
            flag_tmp.push_back(i);
          }
        }

        if (!flag_tmp.empty()) {
          // matlab version is: `ismember(2+1, flag_tmp)`, guessing
          // because the index difference between c++ and matlab
          if (ismember(2, flag_tmp)) {
            pix_found.push_back(pixTmp.rowRange(2, 3));
          }
          if (ismember(2, flag_tmp) &&
              ((!ismember(1, flag_tmp) && !pix_holdup1.empty()) ||
               (!ismember(0, flag_tmp) && !pix_holdup2.empty()))) {
            pix_write.push_back(pix_holdup1);
            pix_write.push_back(pix_holdup2);
            pix_holdup1.release();
            pix_holdup2.release();
          }
          if (ismember(1, flag_tmp) || ismember(0, flag_tmp)) {
            if (ismember(1, flag_tmp)) {
              pix_holdup1.push_back(pixTmp.rowRange(1, 2));
            }
            if (ismember(0, flag_tmp)) {
              pix_holdup2.push_back(pixTmp.rowRange(0, 1));
            }
            if (!ismember(2, flag_tmp)) {
              pix_write.push_back(pix_found);
              pix_found = pix_holdup1;
              pix_holdup1 = pix_holdup2;
              pix_holdup2.release();
              pix = pixTmp.rowRange(1, 2);
            }
          }
          if (ismember(2, flag_tmp) &&
              ((!ismember(3, flag_tmp) && !pix_holddown1.empty()) ||
               (!ismember(4, flag_tmp) && !pix_holddown2.empty()))) {
            pix_write.push_back(pix_holddown1);
            pix_write.push_back(pix_holddown2);
            pix_holddown1.release();
            pix_holddown2.release();
          }
          if (ismember(3, flag_tmp) || ismember(4, flag_tmp)) {
            if (ismember(3, flag_tmp)) {
              pix_holddown1.push_back(pixTmp.rowRange(3, 4));
            }
            if (ismember(4, flag_tmp)) {
              pix_holddown2.push_back(pixTmp.rowRange(4, 5));
            }
            if (!ismember(2, flag_tmp)) {
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

  // save pix_write to .csv
  // const std::string check =
  //     "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
  //     "output/pix_write.csv";
  // write_csv(check, pix_write);
  std::vector<int> prop_idx;
  for (int i = 0; i < pix_write.rows; i++) {
    const double* pix_ptr = pix_write.ptr<double>(i);
    if (pix_ptr[0] >= 0 && pix_ptr[0] < nc && pix_ptr[1] >= 0 &&
        pix_ptr[1] < nr) {
      prop_idx.push_back(
          sub2ind_along_y(nr, nc, (int)pix_ptr[1], (int)pix_ptr[0]));
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
  const int rows0 = xy_orig_tmp.rows;
  const int cols0 = xy_orig_tmp.cols;
  xy_orig = cv::Mat::zeros(rows0, cols0, CV_64F);
  for (int row = 0; row < rows0; row++) {
    const double* ptr = xy_orig_tmp.ptr<double>(row);
    for (int col = 0; col < cols0; col++) {
      xy_orig.ptr<double>(row)[col] = num2fix_unsigned(ptr[col] / scale, 9);
    }
  }

  // floor elements in xy_orig
  cv::Mat xy_orig_int;
  xy_orig_int = cv::Mat::zeros(rows0, cols0, CV_64F);
  for (int row = 0; row < rows0; row++) {
    const double* ptr = xy_orig.ptr<double>(row);
    for (int col = 0; col < cols0; col++) {
      xy_orig_int.ptr<double>(row)[col] = cvFloor(ptr[col]);
    }
  }
  cv::Mat xy_orig_frac(xy_orig.rows, xy_orig.cols, CV_64F);
  cv::subtract(xy_orig, xy_orig_int, xy_orig_frac);

  cv::Mat final_flag, final_mask, final_xy_orig_int, final_xy_orig_frac;
  final_flag = xy_orig_int.colRange(0, 1) >= 0 &
               xy_orig_int.colRange(0, 1) < int(nc / scale - 1) &
               xy_orig_int.colRange(1, 2) >= 0 &
               xy_orig_int.colRange(1, 2) < int(nr / scale - 1);
  cv::hconcat(final_flag, final_flag, final_mask);
  xy_orig_int.copyTo(final_xy_orig_int, final_mask);
  xy_orig_frac.copyTo(final_xy_orig_frac, final_mask);

  cv::Mat final_idx, prop_idx_mat;
  // convert vector prop_idx to cv::mat prop_idx_mat
  prop_idx_mat = cv::Mat(prop_idx);
  prop_idx_mat.copyTo(final_idx, final_flag);

  std::vector<std::tuple<int, int>> coordinates;
  const int f_row = final_idx.rows;
  coordinates.reserve(f_row);
  for (int i = 0; i < f_row; i++) {
    // use ind2sub here, not ind2sub_along_y
    coordinates.emplace_back(ind2sub(nr, nc, *final_idx.ptr<int>(i)));
  }

  std::vector<int> ind;
  ind.reserve(coordinates.size());
  for (auto [x, y] : coordinates) {
    ind.push_back(sub2ind_along_y(nr / scale, nc / scale, y, x));
  }

  cv::Mat img_rect;
  img_rect = bilinear_remap(img_r, img_g, img_b, final_xy_orig_int,
                            final_xy_orig_frac, ind);

  rect_buffer.rect_img = img_rect;
  return rect_buffer;
}

cv::Mat bilinear_remap(const cv::Mat& img_r, const cv::Mat& img_g,
                       const cv::Mat& img_b, const cv::Mat& final_xy_orig_int,
                       const cv::Mat& final_xy_orig_frac,
                       std::vector<int> ind) {
  const int nr = img_r.rows;
  const int nc = img_r.cols;

  cv::Mat bilinear_pix_floor1, bilinear_pix_floor2, bilinear_pix_floor3,
      bilinear_pix_floor4;

  bilinear_pix_floor1 = final_xy_orig_int;
  cv::hconcat(final_xy_orig_int.colRange(0, 1) + 1,
              final_xy_orig_int.colRange(1, 2), bilinear_pix_floor2);
  cv::hconcat(final_xy_orig_int.colRange(0, 1),
              final_xy_orig_int.colRange(1, 2) + 1, bilinear_pix_floor3);
  cv::hconcat(final_xy_orig_int.colRange(0, 1) + 1,
              final_xy_orig_int.colRange(1, 2) + 1, bilinear_pix_floor4);

  std::vector<int> bilinear_ind1, bilinear_ind2, bilinear_ind3, bilinear_ind4;
  const int row0 = final_xy_orig_int.rows;
  for (int i = 0; i < row0; i++) {
    bilinear_ind1.push_back(
        sub2ind_along_y(nr, nc, (int)bilinear_pix_floor1.ptr<double>(i)[1],
                        (int)bilinear_pix_floor1.ptr<double>(i)[0]));
    bilinear_ind2.push_back(
        sub2ind_along_y(nr, nc, (int)bilinear_pix_floor2.ptr<double>(i)[1],
                        (int)bilinear_pix_floor2.ptr<double>(i)[0]));
    bilinear_ind3.push_back(
        sub2ind_along_y(nr, nc, (int)bilinear_pix_floor3.ptr<double>(i)[1],
                        (int)bilinear_pix_floor3.ptr<double>(i)[0]));
    bilinear_ind4.push_back(
        sub2ind_along_y(nr, nc, (int)bilinear_pix_floor4.ptr<double>(i)[1],
                        (int)bilinear_pix_floor4.ptr<double>(i)[0]));
  }

  cv::Mat coeff1 = (1 - final_xy_orig_frac.colRange(1, 2))
                       .mul((1 - final_xy_orig_frac.colRange(0, 1)));
  cv::Mat coeff2 = (1 - final_xy_orig_frac.colRange(1, 2))
                       .mul(final_xy_orig_frac.colRange(0, 1));
  cv::Mat coeff3 = (final_xy_orig_frac.colRange(1, 2))
                       .mul((1 - final_xy_orig_frac.colRange(0, 1)));
  cv::Mat coeff4 = (final_xy_orig_frac.colRange(1, 2))
                       .mul(final_xy_orig_frac.colRange(0, 1));

  const int row1 = coeff1.rows;
  for (int i = 0; i < row1; i++) {
    coeff1.ptr<double>(i)[0] = num2fix_unsigned(coeff1.ptr<double>(i)[0], 9);
    coeff2.ptr<double>(i)[0] = num2fix_unsigned(coeff2.ptr<double>(i)[0], 9);
    coeff3.ptr<double>(i)[0] = num2fix_unsigned(coeff3.ptr<double>(i)[0], 9);
    coeff4.ptr<double>(i)[0] = num2fix_unsigned(coeff4.ptr<double>(i)[0], 9);
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

  coordinate_generator(coord1_r, coeff1, img_r, bilinear_ind1, 9);
  coordinate_generator(coord2_r, coeff2, img_r, bilinear_ind2, 9);
  coordinate_generator(coord3_r, coeff3, img_r, bilinear_ind3, 9);
  coordinate_generator(coord4_r, coeff4, img_r, bilinear_ind4, 9);

  std::vector<double> coord1_g, coord2_g, coord3_g, coord4_g;

  coordinate_generator(coord1_g, coeff1, img_g, bilinear_ind1, 9);
  coordinate_generator(coord2_g, coeff2, img_g, bilinear_ind2, 9);
  coordinate_generator(coord3_g, coeff3, img_g, bilinear_ind3, 9);
  coordinate_generator(coord4_g, coeff4, img_g, bilinear_ind4, 9);

  std::vector<double> coord1_b, coord2_b, coord3_b, coord4_b;

  coordinate_generator(coord1_b, coeff1, img_b, bilinear_ind1, 9);
  coordinate_generator(coord2_b, coeff2, img_b, bilinear_ind2, 9);
  coordinate_generator(coord3_b, coeff3, img_b, bilinear_ind3, 9);
  coordinate_generator(coord4_b, coeff4, img_b, bilinear_ind4, 9);

  cv::Mat r_rect, g_rect, b_rect;
  r_rect = cv::Mat::zeros(nr, nc, CV_8UC1);
  g_rect = cv::Mat::zeros(nr, nc, CV_8UC1);
  b_rect = cv::Mat::zeros(nr, nc, CV_8UC1);

  const int row2 = (int)ind.size();
  for (int i = 0; i < row2; i++) {
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

void coordinate_generator(std::vector<double>& vec, cv::Mat coeff,
                          cv::Mat img_ch, std::vector<int> index,
                          int frac_len) {
  const int row = coeff.rows;
  for (int i = 0; i < row; i++) {
    vec.emplace_back(num2fix_unsigned(
        coeff.ptr<double>(i)[0] * double(img_ch.at<uint8_t>(index[i])),
        frac_len));
  }
}
