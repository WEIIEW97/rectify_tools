#include "rect_img.h"

#include "lut_parser.h"
#include "utils.h"

cv::Mat rect_img(const cv::Mat& xOrig2Rect, const cv::Mat& yOrig2Rect,
                 cv::Mat xRect2Orig, cv::Mat yRect2Orig, const cv::Mat& image,
                 int scale) {
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
        // int idx = 360;

        cv::Mat pixRect_1;
        cv::Mat x_slice, y_slice;
        //        cv::transpose(xOrig2Rect(cv::Range(y_sampled.at<double>(idx),
        //        y_sampled.at<double>(idx)+scale), cv::Range::all()), x_slice);
        //        cv::transpose(yOrig2Rect(cv::Range(y_sampled.at<double>(idx),
        //        y_sampled.at<double>(idx)+scale), cv::Range::all()), y_slice);
        cv::transpose(xOrig2Rect(cv::Range((int)y_sampled[idx],
                                           (int)y_sampled[idx] + scale),
                                 cv::Range::all()),
                      x_slice);
        cv::transpose(yOrig2Rect(cv::Range((int)y_sampled[idx],
                                           (int)y_sampled[idx] + scale),
                                 cv::Range::all()),
                      y_slice);
        cv::hconcat(x_slice, y_slice, pixRect_1);

        // scale=2 was wrong in original code
        // ceiling operation for every pixRect_1 ???
        for (int i = 0; i < pixRect_1.size[0]; i++) {
            for (int j = 0; j < 2; j++) {  // dimension is (N, 2)
                pixRect_1.at<double>(i, j) = cvCeil(pixRect_1.at<double>(
                    i, j));  // this may not work sice ceil(double num) but not
                             // ceil(int num)
            }
        }
        //        std::cout << pixRect_1 << std::endl;

        cv::Mat flag =
            pixRect_1.colRange(0, 1) >= 0 & pixRect_1.colRange(0, 1) < nc &
            pixRect_1.colRange(1, 2) >= 0 & pixRect_1.colRange(1, 2) < nr;
        cv::Mat mask;
        hconcat(flag, flag, mask);
        //        std::cout << mask << std::endl;
        cv::Mat res;
        pixRect_1.copyTo(res, mask);
        //        std::cout << res << std::endl;
        // take elements which != 0;
        cv::Mat pixRect;
        for (int i = 0; i < flag.rows; i++) {
            if (flag.at<uint8_t>(i, 0) != 0) {
                pixRect.push_back(res.row(i));
            }
        }

        // ??? why would we do this?
        if (!pixRect.empty()) {
            if (int(pixRect.at<double>(0, 0)) % scale != 0) {
                pixRect.at<double>(0, 0) = pixRect.at<double>(0, 0) - 1;
            } else if (int(pixRect.at<double>(0, 1)) % scale != 0) {
                pixRect.at<double>(0, 1) = pixRect.at<double>(0, 1) - 1;
            }
        } else
            continue;

        if (!pixRect.empty()) {
            cv::Mat startPixRect =
                pixRect.rowRange(0, 1)
                    .clone();  // [0, 1) left closed right open.
            //            std::cout << startPixRect << std::endl;
            cv::Mat pix = startPixRect;
            //            std::cout << pix << std::endl;

            //            int type;
            //            type = pix.type();
            //            std::cout << type << std::endl;

            cv::Mat pix_found, pix_holdup1, pix_holdup2, pix_holddown1,
                pix_holddown2;

            for (int j = (int)startPixRect.at<double>(0, 0); j < nc; j++) {
                // for (int j = 637; j < nc; j++) {
                cv::Mat pixTmp_l =
                    cv::Mat::ones(5, 1, CV_64F) * j;  // repmat(j, 5, 1)
                cv::Mat pixTmp_r(5, 1, CV_64F);
                pixTmp_r.at<double>(0, 0) = pix.at<double>(0, 1) - 2 * scale;
                pixTmp_r.at<double>(1, 0) = pix.at<double>(0, 1) - 1 * scale;
                pixTmp_r.at<double>(2, 0) = pix.at<double>(0, 1);
                pixTmp_r.at<double>(3, 0) = pix.at<double>(0, 1) + 1 * scale;
                pixTmp_r.at<double>(4, 0) = pix.at<double>(0, 1) + 2 * scale;

                // test
                //                std::cout << pix << std::endl;
                cv::Mat pixTmp;
                cv::hconcat(pixTmp_l, pixTmp_r, pixTmp);

                //                std::cout << pixTmp << std::endl;

                cv::Mat flagIn(5, 1, CV_64F);
                flagIn =
                    pixTmp.colRange(0, 1) >= 0 & pixTmp.colRange(0, 1) < nc &
                    pixTmp.colRange(1, 2) >= 0 & pixTmp.colRange(1, 2) < nr;
                //                std::cout << flagIn << std::endl;
                cv::Mat mask_height, mask_width;
                pixTmp.colRange(1, 2).copyTo(mask_height, flagIn);
                pixTmp.colRange(0, 1).copyTo(mask_width, flagIn);

                // convert non-zero element to vector for mask_height and
                // mask_width
                cv::Mat pixTmp_height, pixTmp_width, index;

                for (int i = 0; i < flagIn.rows; i++) {
                    if (flagIn.at<uint8_t>(i, 0) != 0) {
                        pixTmp_height.push_back(mask_height.row(i));
                        pixTmp_width.push_back(mask_width.row(i));
                    }
                }

                for (int i = 0; i < pixTmp_width.rows; i++) {
                    index.push_back(sub2ind_along_y(
                        nr, nc, (int)pixTmp_height.at<double>(i),
                        (int)pixTmp_width.at<double>(i)));
                }

                //                cv::Mat pixTmpAddOn_l = cv::Mat::ones(7, 1,
                //                CV_64F) * j; cv::Mat pixTmpAddOn_r(7, 1,
                //                CV_64F); pixTmpAddOn_r.at<double>(0, 0) =
                //                pix.at<double>(0, 1) - 3;
                //                pixTmpAddOn_r.at<double>(1, 0) =
                //                pix.at<double>(0, 1) - 2;
                //                pixTmpAddOn_r.at<double>(2, 0) =
                //                pix.at<double>(0, 1) - 1;
                //                pixTmpAddOn_r.at<double>(3, 0) =
                //                pix.at<double>(0, 1);
                //                pixTmpAddOn_r.at<double>(4, 0) =
                //                pix.at<double>(0, 1) + 1;
                //                pixTmpAddOn_r.at<double>(5, 0) =
                //                pix.at<double>(0, 1) + 2;
                //                pixTmpAddOn_r.at<double>(6, 0) =
                //                pix.at<double>(0, 1) + 3;
                //
                //                cv::Mat pixTmpAddOn;
                //                cv::hconcat(pixTmpAddOn_l, pixTmpAddOn_r,
                //                pixTmpAddOn);
                //
                //                cv::Mat flagIn2(7, 1, CV_64F);
                //                flagIn2 = pixTmpAddOn.colRange(0, 1) >= 0 &
                //                          pixTmpAddOn.colRange(0, 1) < nc &
                //                          pixTmpAddOn.colRange(1, 2) >= 0 &
                //                          pixTmpAddOn.colRange(1, 2) < nr;
                //                cv::Mat mask_height2, mask_width2;
                //                pixTmpAddOn.colRange(1,
                //                2).copyTo(mask_height2, flagIn2);
                //                pixTmpAddOn.colRange(0, 1).copyTo(mask_width2,
                //                flagIn2);
                //
                //                cv::Mat pixTmpAddOn_height, pixTmpAddOn_width,
                //                index2; for (int i = 0; i < mask_height2.rows;
                //                i++) {
                //                    if (mask_height2.at<double>(i, 0) != 0) {
                //                        pixTmpAddOn_height.push_back(mask_height2.row(i));
                //                    }
                //                }
                //                for (int i = 0; i < mask_width2.rows; i++) {
                //                    if (mask_width2.at<double>(i, 0) != 0) {
                //                        pixTmpAddOn_width.push_back(mask_width2.row(i));
                //                    }
                //                }
                //                for (int i = 0; i < pixTmpAddOn_width.rows;
                //                i++) {
                //                    index2.push_back(sub2ind_along_y(
                //                        nr, nc,
                //                        (int)pixTmpAddOn_height.at<double>(i),
                //                        (int)pixTmpAddOn_width.at<double>(i)));
                //                }
                //
                //                cv::Mat buffer_x, buffer_y, buffer;
                //                for (int i = 0; i < index2.rows; i++) {
                //                    buffer_x.push_back(
                //                        xRect2Orig.at<double>(index2.at<int>(i)));
                //                    buffer_y.push_back(
                //                        yRect2Orig.at<double>(index2.at<int>(i)));
                //                }
                //                cv::hconcat(buffer_x, buffer_y, buffer);
                //                std::cout << buffer << std::endl;

                cv::Mat bufferTmp_x, bufferTmp_y, rect2orig_tmp_in;
                for (int i = 0; i < index.rows; i++) {
                    bufferTmp_x.push_back(
                        xRect2Orig.at<double>(index.at<int>(i)));
                    bufferTmp_y.push_back(
                        yRect2Orig.at<double>(index.at<int>(i)));
                }
                cv::hconcat(bufferTmp_x, bufferTmp_y, rect2orig_tmp_in);
                cv::Mat rect2orig_tmp =
                    cv::Mat::ones(pixTmp.rows, pixTmp.cols, CV_64F) * -10;
                // cv::Mat rect2orig_mask, rect2orig_buffer;
                // cv::hconcat(flagIn, flagIn, rect2orig_mask);

                std::vector<int> flag_index;
                for (int i = 0; i < flagIn.rows; i++) {
                    if (flagIn.at<uint8_t>(i, 0) != 0) {
                        flag_index.push_back(i);
                    }
                }

                //                std::cout << flagIn << std::endl;

                // rect2orig_tmp.copyTo(rect2orig_buffer, rect2orig_mask);
                // rect2orig_buffer = rect2orig_tmp_in;

                //  std::cout << "rect2orig_tmp: " << rect2orig_tmp <<
                //  std::endl; std::cout << "rect2orig_tmp_in: " <<
                //  rect2orig_tmp_in << std::endl; std::cout <<
                //  "rect2orig_buffer: " << rect2orig_buffer << std::endl;

                // for (int row = flag_index[0]; row < rect2orig_tmp.rows;
                // row++) {
                //     for (int col = 0; col < rect2orig_tmp.cols; col++) {
                //         rect2orig_tmp.at<double>(row, col) =
                //             rect2orig_tmp_in.at<double>(row - flag_index[0],
                //                                         col);
                //     }
                // }

                for (int row = 0; row < rect2orig_tmp.rows; row++) {
                    for (int by = 0; by < flag_index.size(); by++) {
                        if (row == flag_index[by]) {
                            rect2orig_tmp.at<double>(row, 0) =
                                rect2orig_tmp_in.at<double>(by, 0);
                            rect2orig_tmp.at<double>(row, 1) =
                                rect2orig_tmp_in.at<double>(by, 1);
                        }
                    }
                }

                //                std::cout << rect2orig_tmp << std::endl;

                // cv::Mat zero_flag = cv::Mat::zeros(5, 1, CV_64F);
                cv::Mat _rect2orig_tmp(rect2orig_tmp.rows, rect2orig_tmp.cols,
                                       CV_64F);
                for (int row = 0; row < rect2orig_tmp.rows; row++) {
                    for (int col = 0; col < rect2orig_tmp.cols; col++) {
                        _rect2orig_tmp.at<double>(row, col) = num2fix_unsigned(
                            rect2orig_tmp.at<double>(row, col) / scale, 9);
                    }
                }

                //                std::cout << _rect2orig_tmp << std::endl;

                std::vector<double> flag_tmp;
                for (int i = 0; i < _rect2orig_tmp.rows; i++) {
                    if (cvFloor(_rect2orig_tmp.at<double>(i, 0)) >= 0 &&
                        cvFloor(_rect2orig_tmp.at<double>(i, 0)) <=
                            (nc / scale - 2) &&
                        rect2orig_tmp.at<double>(i, 1) >= y_sampled[idx] &&
                        rect2orig_tmp.at<double>(i, 1) < y_sampled[idx + 1] &&
                        rect2orig_tmp.at<double>(i, 1) >= 0 &&
                        rect2orig_tmp.at<double>(i, 1) < nr &&
                        pixTmp.at<double>(i, 1) >= 0 &&
                        pixTmp.at<double>(i, 1) < nr) {
                        flag_tmp.push_back(i);
                    }
                }

                if (!flag_tmp.empty()) {
                    // matlab version is: `ismember(2+1, flag_tmp)`, guessing
                    // because the index difference between c++ and matlab
                    //                    std::cout << _rect2orig_tmp <<
                    //                    std::endl;
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
            // std::cout << "pix_found size: " << pix_found.size() << std::endl;
        } else
            continue;
    }

    // std::cout << pix_write << std::endl;

    // test: exclude coordinates where < 5
    //    cv::Mat pix_write_test;
    //    for (int i = 0; i < pix_write.rows; i++) {
    //        if (pix_write.at<double>(i, 0) >= 5) {
    //            pix_write_test.push_back(pix_write.row(i));
    //        }
    //    }
    //
    //    std::cout << pix_write_test << std::endl;

    // test: try pix_write_test intead of pix_write.
    // check row numbers.

    // save pix_write to .csv
    // const std::string check =
    //     "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
    //     "output/pix_write.csv";
    // write_csv(check, pix_write);
    std::vector<int> prop_idx;
    for (int i = 0; i < pix_write.rows; i++) {
        if (pix_write.at<double>(i, 0) >= 0 &&
            pix_write.at<double>(i, 0) < nc &&
            pix_write.at<double>(i, 1) >= 0 &&
            pix_write.at<double>(i, 1) < nr) {
            prop_idx.push_back(
                sub2ind_along_y(nr, nc, (int)pix_write.at<double>(i, 1),
                                (int)pix_write.at<double>(i, 0)));
        }
    }

    cv::Mat x_orig, y_orig, xy_orig_tmp;
    for (int i : prop_idx) {
        x_orig.push_back(xRect2Orig.at<double>(i));
        y_orig.push_back(yRect2Orig.at<double>(i));
    }
    cv::hconcat(x_orig, y_orig, xy_orig_tmp);

    cv::Mat xy_orig;
    xy_orig = cv::Mat::zeros(xy_orig_tmp.rows, xy_orig_tmp.cols, CV_64F);
    for (int row = 0; row < xy_orig_tmp.rows; row++) {
        for (int col = 0; col < xy_orig_tmp.cols; col++) {
            xy_orig.at<double>(row, col) =
                num2fix_unsigned(xy_orig_tmp.at<double>(row, col) / scale, 9);
        }
    }

    // floor elements in xy_orig
    cv::Mat xy_orig_int;
    xy_orig_int = cv::Mat::zeros(xy_orig.rows, xy_orig.cols, CV_64F);
    for (int row = 0; row < xy_orig.rows; row++) {
        for (int col = 0; col < xy_orig.cols; col++) {
            xy_orig_int.at<double>(row, col) =
                cvFloor(xy_orig.at<double>(row, col));
        }
    }
    cv::Mat xy_orig_frac(xy_orig.rows, xy_orig.cols, CV_64F);
    cv::subtract(xy_orig, xy_orig_int, xy_orig_frac);

    //    std::cout << xy_orig_frac << std::endl;

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
    coordinates.reserve(final_idx.rows);
    for (int i = 0; i < final_idx.rows; i++) {
        // use ind2sub here, not ind2sub_along_y
        coordinates.emplace_back(ind2sub(nr, nc, final_idx.at<int>(i)));
    }

    std::vector<int> ind;
    ind.reserve(coordinates.size());
    for (auto [x, y] : coordinates) {
        ind.push_back(sub2ind_along_y(nr / scale, nc / scale, y, x));
    }

    cv::Mat img_rect;
    img_rect = bilinear_remap(img_r, img_g, img_b, final_xy_orig_int,
                              final_xy_orig_frac, ind);

    //    cv::Scalar rect_sum = sum(img_rect);
    //    std::cout << rect_sum << std::endl;

    return img_rect;
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
    for (int i = 0; i < final_xy_orig_int.rows; i++) {
        bilinear_ind1.push_back(
            sub2ind_along_y(nr, nc, (int)bilinear_pix_floor1.at<double>(i, 1),
                            (int)bilinear_pix_floor1.at<double>(i, 0)));
    }
    for (int i = 0; i < final_xy_orig_int.rows; i++) {
        bilinear_ind2.push_back(
            sub2ind_along_y(nr, nc, (int)bilinear_pix_floor2.at<double>(i, 1),
                            (int)bilinear_pix_floor2.at<double>(i, 0)));
    }
    for (int i = 0; i < final_xy_orig_int.rows; i++) {
        bilinear_ind3.push_back(
            sub2ind_along_y(nr, nc, (int)bilinear_pix_floor3.at<double>(i, 1),
                            (int)bilinear_pix_floor3.at<double>(i, 0)));
    }
    for (int i = 0; i < final_xy_orig_int.rows; i++) {
        bilinear_ind4.push_back(
            sub2ind_along_y(nr, nc, (int)bilinear_pix_floor4.at<double>(i, 1),
                            (int)bilinear_pix_floor4.at<double>(i, 0)));
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
        coeff1.at<double>(i, 0) = num2fix_unsigned(coeff1.at<double>(i, 0), 9);
    }
    for (int i = 0; i < coeff2.rows; i++) {
        coeff2.at<double>(i, 0) = num2fix_unsigned(coeff2.at<double>(i, 0), 9);
    }
    for (int i = 0; i < coeff3.rows; i++) {
        coeff3.at<double>(i, 0) = num2fix_unsigned(coeff3.at<double>(i, 0), 9);
    }
    for (int i = 0; i < coeff4.rows; i++) {
        coeff4.at<double>(i, 0) = num2fix_unsigned(coeff4.at<double>(i, 0), 9);
    }

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

void coordinate_generator(std::vector<double>& vec, cv::Mat coeff,
                          cv::Mat img_ch, std::vector<int> index,
                          int frac_len) {
    for (int i = 0; i < coeff.rows; i++) {
        vec.emplace_back(num2fix_unsigned(
            coeff.at<double>(i, 0) * double(img_ch.at<uint8_t>(index[i])),
            frac_len));
    }
}