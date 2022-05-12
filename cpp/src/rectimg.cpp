#include "rectImg.h"
#include "utils.h"
#include "lutParser.h"


cv::Mat rect_img(const cv::Mat& xOrig2Rect, const cv::Mat& yOrig2Rect, cv::Mat xRect2Orig, cv::Mat yRect2Orig, const cv::Mat& image, int scale) {
    cv::Mat img_rect = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);

    const int nc = scale * image.rows;
    const int nr = scale * image.cols;

    std::vector<boost::any> pix_write;

    cv::Mat mesh_x, mesh_y;
    meshgrid(cv::Range(1, nc), cv::Range(1, nr), mesh_x, mesh_y);

    int y_size = (nr-scale)/scale + 1;
    // cv::Mat y_sampled(y_size, 1, CV_64F);
    cv::Mat y_sampled;
    std::vector<double> y_samp_vec;
    for (int i=0; i<y_size; i++) {
        y_samp_vec.push_back(scale + scale * i);
        if (y_samp_vec.back() != nr) {
            // cv::Mat y_sample(y_size+1, 1, CV_64F);
            // for (int i=0; i<y_size; i++) {
            //     y_sample.at<double>(i, 0) = y_sampled.at<double>(i, 0);
            // }
            // y_sample.at<double>(y_sample.size[0]-1, 0) = nr;
            y_samp_vec.push_back(nr);
        }
    }
    y_sampled = cv::Mat(y_samp_vec);

    // TODO: add crop here

    /* retrieve BGR channel separately */
    cv::Mat img_r, img_g, img_b;
    cv::extractChannel(image, img_r, 2);
    cv::extractChannel(image, img_g, 1);
    cv::extractChannel(image, img_b, 0);

    for (int ii=scale; ii<scale*(nc-1); ii+=scale) {
        int idx = ii / scale;

        cv::Mat pixRect_1;
        cv::Mat x_slice, y_slice;
        cv::transpose(xOrig2Rect(cv::Range(y_sampled.at<int>(idx), y_sampled.at<int>(idx)+scale-1), cv::Range::all()), x_slice);
        cv::transpose(yOrig2Rect(cv::Range(y_sampled.at<int>(idx), y_sampled.at<int>(idx)+scale-1), cv::Range::all()), y_slice);
        cv::hconcat(x_slice, y_slice, pixRect_1);
        
        // scale=2 was wrong in original code
        // ceiling operation for every pixRect_1 ???
        for (int i=0; i<pixRect_1.size[0]; i++) {
            for (int j=0; j<2; j++) {   // dimension is (N, 2)
                pixRect_1.at<double>(i, j) = cvCeil(pixRect_1.at<double>(i, j)); // this may not work sice ceil(double num) but not ceil(int num)
            }
        }

        cv::Mat flag = pixRect_1.colRange(0,1) >= 0 & pixRect_1.colRange(0,1) <= nc & pixRect_1.colRange(1,2) >= 0 & pixRect_1.colRange(1,2) <= nr;
        cv::Mat mask;
        hconcat(flag, flag, mask);
        cv::Mat res;
        pixRect_1.copyTo(res, flag);
        // take elements which != 0;
        cv::Mat pixRect;
        for (int i=0; i<res.rows; i++) {
            if (res.at<double>(i, 0) != 0) {
                pixRect.push_back(res.row(i));
            }
        }

        // ??? why would we do this?
        if (!pixRect.empty()) {
            if (int(pixRect.at<double>(0, 0)) % scale != 0) {
                pixRect.at<double>(0, 0) = pixRect.at<double>(0, 0) - 1;
            }
            else if (int(pixRect.at<double>(0, 1)) % scale != 0) {
                pixRect.at<double>(0 ,1) = pixRect.at<double>(0 ,1) - 1;
            }
        }

        if (!pixRect.empty()) {
            cv::Mat startPixRect = pixRect.rowRange(0, 1).clone(); // [0, 1) left closed right open.
            cv::Mat pix = startPixRect;

            std::vector<boost::any> pix_found, pix_holdup1, pix_holdup2, pix_holddown1, pix_holddown2;

            for (int j=startPixRect.at<int>(0, 0); j < nc; j++) {
                cv::Mat pixTmp_l = cv::Mat::ones(5, 1, CV_64F) * j;  // repmat(j, 5, 1)
                cv::Mat pixTmp_r(5, 1, CV_64F);
                pixTmp_r.at<double>(0, 0) = pix.at<double>(0, 1) - 2;
                pixTmp_r.at<double>(1, 0) = pix.at<double>(0, 1) - 1;
                pixTmp_r.at<double>(2, 0) = pix.at<double>(0, 1);
                pixTmp_r.at<double>(3, 0) = pix.at<double>(0, 1) + 1;
                pixTmp_r.at<double>(4, 0) = pix.at<double>(0, 1) + 2;

                cv::Mat pixTmp;
                cv::hconcat(pixTmp_l, pixTmp_r, pixTmp);

                cv::Mat flagIn(5, 1, CV_64F);
                flagIn = pixTmp.colRange(0, 1) >= 0 & pixTmp.colRange(0, 1) <= nc & pixTmp.colRange(1, 2) >= 0 & pixTmp.colRange(1, 2) <= nr;
                cv::Mat mask_height, mask_width;
                pixTmp.colRange(1, 2).copyTo(flagIn, mask_height);
                pixTmp.colRange(0, 1).copyTo(flagIn, mask_width);
                // convert non-zero element to vector for mask_height and mask_width
                cv::Mat pixTmp_height, pixTmp_width, index;
                for (int i=0; i<mask_height.rows; i++) {
                    if (mask_height.at<double>(i, 0) != 0) {
                        pixTmp_height.push_back(mask_height.row(i));
                    }
                }
                for (int i=0; i<mask_width.rows; i++) {
                    if (mask_width.at<double>(i, 1) != 0) {
                        pixTmp_width.push_back(mask_width.row(i));
                    }
                }
                for (int i=0; i<pixTmp_width.rows; i++) {
                    index.push_back(sub2ind(nr, nc, (int)pixTmp_width.at<double>(i), (int)pixTmp_height.at<double>(i)));
                }

                cv::Mat pixTmpAddOn_l = cv::Mat::ones(7, 1, CV_64F) * j;
                cv::Mat pixTmpAddOn_r(7, 1, CV_64F);
                pixTmpAddOn_r.at<double>(0, 0) = pix.at<double>(0, 1) - 3;
                pixTmpAddOn_r.at<double>(1, 0) = pix.at<double>(0, 1) - 2;
                pixTmpAddOn_r.at<double>(2, 0) = pix.at<double>(0, 1) - 1;
                pixTmpAddOn_r.at<double>(3, 0) = pix.at<double>(0, 1);
                pixTmpAddOn_r.at<double>(4, 0) = pix.at<double>(0, 1) + 1;
                pixTmpAddOn_r.at<double>(5, 0) = pix.at<double>(0, 1) + 2;
                pixTmpAddOn_r.at<double>(6, 0) = pix.at<double>(0, 1) + 3;

                cv::Mat pixTmpAddOn;
                cv::hconcat(pixTmpAddOn_l, pixTmpAddOn_r, pixTmpAddOn);

                cv::Mat flagIn2(7, 1, CV_64F);
                flagIn2 = pixTmpAddOn.colRange(0, 1) >= 0 & pixTmpAddOn.colRange(0, 1) <= nc & pixTmpAddOn.colRange(1, 2) >= 0 & pixTmpAddOn.colRange(1, 2) <= nr;
                cv::Mat mask_height2, mask_width2;
                pixTmpAddOn.colRange(1, 2).copyTo(flagIn2, mask_height2);
                pixTmpAddOn.colRange(0, 1).copyTo(flagIn2, mask_width2);

                cv::Mat pixTmpAddOn_height, pixTmpAddOn_width, index2;
                for (int i=0; i<mask_height2.rows; i++) {
                    if (mask_height2.at<double>(i, 0) != 0) {
                        pixTmpAddOn_height.push_back(mask_height2.row(i));
                    }
                }
                for (int i=0; i<mask_width2.rows; i++) {
                    if (mask_width2.at<double>(i, 1) != 0) {
                        pixTmpAddOn_width.push_back(mask_width2.row(i));
                    }
                }
                for (int i=0; i<pixTmpAddOn_width.rows; i++) {
                    index2.push_back(sub2ind(nr, nc, (int)pixTmpAddOn_width.at<double>(i), (int)pixTmpAddOn_height.at<double>(i)));
                }
                cv::Mat buffer_x, buffer_y, buffer;
                for (int i=0; i<index2.rows; i++) {
                    buffer_x.push_back(xRect2Orig.at<double>(index2.at<int>(i)));
                    buffer_y.push_back(xRect2Orig.at<double>(index2.at<int>(i)));
                }
                cv::hconcat(buffer_x, buffer_y, buffer);

                cv::Mat bufferTmp_x, bufferTmp_y, rect2orig_tmp_in;
                for (int i=0; i<index.rows; i++) {
                    bufferTmp_x.push_back(xRect2Orig.at<double>(index.at<int>(i)));
                    bufferTmp_y.push_back(xRect2Orig.at<double>(index.at<int>(i)));
                }
                cv::hconcat(bufferTmp_x, bufferTmp_y, rect2orig_tmp_in);
                cv::Mat rect2orig_tmp = cv::Mat::ones(pixTmp.rows, pixTmp.cols, CV_64F) * -10;
                cv::Mat rect2orig_mask, rect2orig_buffer;
                cv::hconcat(flagIn, flagIn, rect2orig_mask);
                rect2orig_tmp.copyTo(rect2orig_mask, rect2orig_buffer);
                rect2orig_buffer = rect2orig_tmp_in;

                // cv::Mat zero_flag = cv::Mat::zeros(5, 1, CV_64F);
                cv::Mat _rect2orig_tmp(rect2orig_tmp.rows, rect2orig_tmp.cols, CV_64F);
                for (int row=0; row<rect2orig_tmp.rows; row++) {
                    for (int col=0; col<rect2orig_tmp.cols; col++) {
                        _rect2orig_tmp.at<double>(row, col) = num2fix(rect2orig_tmp.at<double>(row, col) / scale, 9);
                    }
                }
                std::vector<double> flag_tmp;
                for (int i=0; i<rect2orig_tmp.rows; i++) {
                    if (cvFloor(_rect2orig_tmp.at<double>(i, 0))>=1 & cvFloor(_rect2orig_tmp.at<double>(i, 0))<=(nc/scale-1) & rect2orig_tmp.at<double>(i, 1)>=y_sampled.at<double>(idx) \
                        & rect2orig_tmp.at<double>(i, 1)<y_sampled.at<double>(idx+1) & rect2orig_tmp.at<double>(i, 1)>0 & rect2orig_tmp.at<double>(i, 1)<=nr & pixTmp.at<double>(i, 1)>0 & pixTmp.at<double>(i, 1)<=nr) {
                        flag_tmp.push_back(i);
                    }
                }

                if (!flag_tmp.empty()) {
                    // matlab version is: `ismember(2+3, flag_tmp)`, guessing because the index difference between c++ and matlab
                    if (ismember(2, flag_tmp)) {
                        pix_found.emplace_back(pixTmp.rowRange(2,3));
                    }
                    if (ismember(2, flag_tmp) && ((!ismember(1, flag_tmp) && !pix_holdup1.empty()) || ((!ismember(0, flag_tmp) && !pix_holdup2.empty())))) {
                        pix_write.emplace_back(pix_holdup1);
                        pix_write.emplace_back(pix_holdup2);
                        pix_holdup1.clear();
                        pix_holdup2.clear();
                    }
                    if (ismember(1, flag_tmp) || ismember(0, flag_tmp)) {
                        if (ismember(1, flag_tmp)) {
                            pix_holdup1.emplace_back(pixTmp.rowRange(1,2));
                        }
                        if (ismember(0, flag_tmp)) {
                            pix_holdup2.emplace_back(pixTmp.rowRange(0,1));
                        }
                        if (!ismember(2, flag_tmp)) {
                            pix_write.emplace_back(pix_found);
                            pix_found = pix_holdup1;
                            pix_holdup1 = pix_holdup2;
                            pix_holdup2.clear();
                            pix = pixTmp.rowRange(1,2);
                        }
                    }
                    if (ismember(2, flag_tmp) && ((!ismember(3, flag_tmp) && !pix_holddown1.empty()) || ((!ismember(4, flag_tmp) && !pix_holddown2.empty())))) {
                        pix_write.emplace_back(pix_holddown1);
                        pix_write.emplace_back(pix_holddown2);
                        pix_holddown1.clear();
                        pix_holddown2.clear();
                        }
                    if (ismember(3, flag_tmp) || ismember(4, flag_tmp)) {
                        if (ismember(3, flag_tmp)) {
                            pix_holddown1.emplace_back(pixTmp.rowRange(3,4));
                        }
                        if (ismember(4, flag_tmp)) {
                            pix_holddown2.emplace_back(pixTmp.rowRange(4,5));
                        }
                        if (!ismember(2, flag_tmp)) {
                            pix_write.emplace_back(pix_found);
                            pix_found = pix_holddown1;
                            pix_holddown1 = pix_holddown2;
                            pix_holddown2.clear();
                            pix = pixTmp.rowRange(3,4);
                        }
                    }
                }
            }
            // pix_left means pixels which are not being handled yet. Not the left-hand side left.
            std::vector<boost::any> pix_left;
            pix_left.emplace_back(pix_holdup1);
            pix_left.emplace_back(pix_holdup2);
            pix_left.emplace_back(pix_holddown1);
            pix_left.emplace_back(pix_holddown2);

            pix_write.emplace_back(pix_found);
            if (!pix_left.empty()) {
                pix_write.emplace_back(pix_left);
            }
        }
    }
    // convert boost_any vector to cv::Mat
    cv::Mat pixWrite;
    for (auto &i : pix_write) {
        pixWrite.push_back(boost::any_cast<cv::Mat>(i));
    }

    std::vector<int> prop_idx;
    for (int i=0; i<pixWrite.rows; i++) {
        if (pixWrite.at<double>(i, 0)>=0 && pixWrite.at<double>(i, 0)<=nc && pixWrite.at<double>(i, 1)>=0 && pixWrite.at<double>(i, 1)<=nr) {
            prop_idx.push_back(sub2ind(nr, nc, (int)pixWrite.at<double>(i, 1), (int)pixWrite.at<double>(i, 0)));
        }
    }

    cv::Mat x_orig, y_orig, xy_orig, xy_orig_tmp, xy_orig_int, xy_orig_frac;
    for (int i : prop_idx) {
        x_orig.push_back(xRect2Orig.at<double>(i));
        y_orig.push_back(yRect2Orig.at<double>(i));
    }
    cv::hconcat(x_orig, y_orig, xy_orig_tmp);

    for (int row=0; row<xy_orig_tmp.rows; row++) {
        for (int col=0; col<xy_orig_tmp.cols; col++) {
            xy_orig.at<double>(row, col) = num2fix(xy_orig_tmp.at<double>(row, col) / scale, 9);
        }
    }

    // floor elements in xy_orig
    for (int row=0; row<xy_orig.rows; row++) {
        for (int col = 0; col < xy_orig.cols; col++) {
            xy_orig_int.at<double>(row, col) = cvFloor(xy_orig.at<double>(row, col));
        }
    }

    cv::subtract(xy_orig, xy_orig_int, xy_orig_frac);

    cv::Mat final_flag, final_mask, final_xy_orig_int, final_xy_orig_frac;
    final_flag = xy_orig_int.colRange(0, 1)>=0 & xy_orig_int.colRange(0, 1)<=(nc/scale-1) & xy_orig_int.colRange(1, 2)>=0 & xy_orig_int.colRange(1, 2)<=(nr/scale-1);
    cv::hconcat(final_flag, final_flag, final_mask);
    xy_orig_int.copyTo(final_mask, final_xy_orig_int);
    xy_orig_frac.copyTo(final_mask, final_xy_orig_frac);
    
    cv::Mat final_idx, prop_idx_mat;
    // convert vector prop_idx to cv::mat prop_idx_mat
    prop_idx_mat = cv::Mat(prop_idx);
    prop_idx_mat.copyTo(final_flag, final_idx);

    std::vector<boost::any> coordinates;
    coordinates.reserve(final_idx.rows);
    for (int i=0; i<final_idx.rows; i++) {
        coordinates.emplace_back(ind2sub(nr, nc, final_idx.at<int>(i)));
    }

    std::vector<int> ind;
    ind.reserve(coordinates.size());
    for (auto & coordinate : coordinates) {
        ind.push_back(sub2ind(nr/scale, nc/scale, boost::any_cast<cv::Mat>(coordinate).at<int>(0)/scale, boost::any_cast<cv::Mat>(coordinate).at<int>(1)/scale));
    }

    img_rect = bilinear_remap(img_rect, img_r, img_g, img_b, final_xy_orig_int, final_xy_orig_frac, ind);

    return img_rect;
}


cv::Mat bilinear_remap(cv::Mat& img_rect, cv::Mat img_r, cv::Mat img_g, cv::Mat img_b, cv::Mat final_xy_orig_int, cv::Mat final_xy_orig_frac, std::vector<int> ind) {

    const int nc = img_r.rows;
    const int nr = img_r.cols;
    const int num_pix = nc * nr;

    cv::Mat bilinear_pix_floor1 = final_xy_orig_int;
    cv::Mat bilinear_pix_floor2, bilinear_pix_floor3, bilinear_pix_floor4;

    cv::hconcat(final_xy_orig_int.colRange(0, 1)+1, final_xy_orig_int.colRange(1, 2), bilinear_pix_floor2);
    cv::hconcat(final_xy_orig_int.colRange(0, 1), final_xy_orig_int.colRange(1, 2)+1, bilinear_pix_floor3);
    cv::hconcat(final_xy_orig_int.colRange(0, 1)+1, final_xy_orig_int.colRange(1, 2)+1, bilinear_pix_floor4);

    std::vector<int> bilinear_ind1, bilinear_ind2, bilinear_ind3, bilinear_ind4;
    for (int i=0; i<final_xy_orig_int.rows; i++) {
        bilinear_ind1.push_back(sub2ind(nr, nc, bilinear_pix_floor1.at<int>(i, 1), bilinear_pix_floor1.at<int>(i, 0)));
    }
    for (int i=0; i<final_xy_orig_int.rows; i++) {
        bilinear_ind2.push_back(sub2ind(nr, nc, bilinear_pix_floor2.at<int>(i, 1), bilinear_pix_floor2.at<int>(i, 0)));
    }
    for (int i=0; i<final_xy_orig_int.rows; i++) {
        bilinear_ind3.push_back(sub2ind(nr, nc, bilinear_pix_floor3.at<int>(i, 1), bilinear_pix_floor3.at<int>(i, 0)));
    }
    for (int i=0; i<final_xy_orig_int.rows; i++) {
        bilinear_ind4.push_back(sub2ind(nr, nc, bilinear_pix_floor4.at<int>(i, 1), bilinear_pix_floor4.at<int>(i, 0)));
    }

    cv::Mat coeff1 = (1-final_xy_orig_frac.colRange(1, 2)).mul((1-final_xy_orig_frac.colRange(0, 1)));
    cv::Mat coeff2 = (1-final_xy_orig_frac.colRange(1, 2)).mul(final_xy_orig_frac.colRange(0, 1));
    cv::Mat coeff3 = (final_xy_orig_frac.colRange(1, 2)).mul((1-final_xy_orig_frac.colRange(0, 1)));
    cv::Mat coeff4 = (final_xy_orig_frac.colRange(1, 2)).mul(final_xy_orig_frac.colRange(0, 1));

    for (int i=0; i<coeff1.rows; i++) {
        coeff1.at<double>(i, 0) = num2fix(coeff1.at<double>(i, 0), 9);
    }
    for (int i=0; i<coeff2.rows; i++) {
        coeff2.at<double>(i, 0) = num2fix(coeff2.at<double>(i, 0), 9);
    }
    for (int i=0; i<coeff3.rows; i++) {
        coeff3.at<double>(i, 0) = num2fix(coeff3.at<double>(i, 0), 9);
    }
    for (int i=0; i<coeff4.rows; i++) {
        coeff4.at<double>(i, 0) = num2fix(coeff4.at<double>(i, 0), 9);
    }
    
    std::vector<double> coord1_r, coord2_r, coord3_r, coord4_r;
//    for (int i=0; i<coeff1.rows; i++) {
//        coord1_r.emplace_back(num2fix(coeff1.at<double>(i, 0) * double(img_r.at<uint8_t>(bilinear_ind1[i])), 9));
//    }
//    for (int i=0; i<coeff2.rows; i++) {
//        coord2_r.emplace_back(num2fix(coeff2.at<double>(i, 0) * double(img_r.at<uint8_t>(bilinear_ind2[i])), 9));
//    }
//    for (int i=0; i<coeff3.rows; i++) {
//        coord3_r.emplace_back(num2fix(coeff3.at<double>(i, 0) * double(img_r.at<uint8_t>(bilinear_ind3[i])), 9));
//    }
//    for (int i=0; i<coeff4.rows; i++) {
//        coord4_r.emplace_back(num2fix(coeff4.at<double>(i, 0) * double(img_r.at<uint8_t>(bilinear_ind4[i])), 9));
//    }
    coordinate_generator(coord1_r, coeff1, img_r, bilinear_ind1, 9);
    coordinate_generator(coord2_r, coeff2, img_r, bilinear_ind2, 9);
    coordinate_generator(coord3_r, coeff3, img_r, bilinear_ind3, 9);
    coordinate_generator(coord4_r, coeff4, img_r, bilinear_ind4, 9);



    std::vector<double> coord1_g, coord2_g, coord3_g, coord4_g;
//    for (int i=0; i<coeff1.rows; i++) {
//        coord1_g.emplace_back(num2fix(coeff1.at<double>(i, 0) * double(img_g.at<uint8_t>(bilinear_ind1[i])), 9));
//    }
//    for (int i=0; i<coeff2.rows; i++) {
//        coord2_g.emplace_back(num2fix(coeff2.at<double>(i, 0) * double(img_g.at<uint8_t>(bilinear_ind2[i])), 9));
//    }
//    for (int i=0; i<coeff3.rows; i++) {
//        coord3_g.emplace_back(num2fix(coeff3.at<double>(i, 0) * double(img_g.at<uint8_t>(bilinear_ind3[i])), 9));
//    }
//    for (int i=0; i<coeff4.rows; i++) {
//        coord4_g.emplace_back(num2fix(coeff4.at<double>(i, 0) * double(img_g.at<uint8_t>(bilinear_ind4[i])), 9));
//    }
    coordinate_generator(coord1_g, coeff1, img_g, bilinear_ind1, 9);
    coordinate_generator(coord2_g, coeff2, img_g, bilinear_ind2, 9);
    coordinate_generator(coord3_g, coeff3, img_g, bilinear_ind3, 9);
    coordinate_generator(coord4_g, coeff4, img_g, bilinear_ind4, 9);



    std::vector<double> coord1_b, coord2_b, coord3_b, coord4_b;
//    for (int i=0; i<coeff1.rows; i++) {
//        coord1_b.emplace_back(num2fix(coeff1.at<double>(i, 0) * double(img_b.at<uint8_t>(bilinear_ind1[i])), 9));
//    }
//    for (int i=0; i<coeff2.rows; i++) {
//        coord2_b.emplace_back(num2fix(coeff2.at<double>(i, 0) * double(img_b.at<uint8_t>(bilinear_ind2[i])), 9));
//    }
//    for (int i=0; i<coeff3.rows; i++) {
//        coord3_b.emplace_back(num2fix(coeff3.at<double>(i, 0) * double(img_b.at<uint8_t>(bilinear_ind3[i])), 9));
//    }
//    for (int i=0; i<coeff4.rows; i++) {
//        coord4_b.emplace_back(num2fix(coeff4.at<double>(i, 0) * double(img_b.at<uint8_t>(bilinear_ind4[i])), 9));
//    }
    coordinate_generator(coord1_b, coeff1, img_b, bilinear_ind1, 9);
    coordinate_generator(coord2_b, coeff2, img_b, bilinear_ind2, 9);
    coordinate_generator(coord3_b, coeff3, img_b, bilinear_ind3, 9);
    coordinate_generator(coord4_b, coeff4, img_b, bilinear_ind4, 9);


    for (int i=0; i<ind.size(); i++) {
        img_rect.at<uint8_t>(ind[i]) = floor(coord1_b[i] + coord2_b[i] + coord3_b[i] + coord4_b[i]);
        img_rect.at<uint8_t>(num_pix+ind[i]) = floor(coord1_g[i] + coord2_g[i] + coord3_g[i] + coord4_g[i]);
        img_rect.at<uint8_t>(2*num_pix+ind[i]) = floor(coord1_r[i] + coord2_r[i] + coord3_r[i] + coord4_r[i]);
    }

    return img_rect;
}


void coordinate_generator(std::vector<double>& vec, cv::Mat coeff, cv::Mat img_ch, std::vector<int> index, int frac_len) {
    for (int i=0; i<coeff.rows; i++) {
        vec.emplace_back(num2fix(coeff.at<double>(i, 0) * double(img_ch.at<uint8_t>(index[i])), frac_len));
    }
}