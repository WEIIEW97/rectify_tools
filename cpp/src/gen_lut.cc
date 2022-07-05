#include "gen_lut.h"

std::vector<unsigned int> make_blendian_data(std::vector<unsigned int> number) {
    size_t i;
    unsigned int num[4];
    for (i = 0; i < number.size(); i++) {
        num[0] = number[i] & MAX_BIT;
        num[1] = (number[i] >> 8) & MAX_BIT;
        num[2] = (number[i] >> 16) & MAX_BIT;
        num[3] = (number[i] >> 24) & MAX_BIT;
        number[i] = (num[0] << 24) | (num[1] << 16) | (num[2] << 8) | num[3];
    }
    return number;
}

cv::Mat apply_distortion(cv::Mat x, cv::Mat k) {
    double cols = x.cols;
    int i;
    cv::Mat r2(1, cols, CV_64FC1), r4(1, cols, CV_64FC1), r6(1, cols, CV_64FC1);
    cv::Mat cdist(1, cols, CV_64FC1), xd1(2, cols, CV_64FC1);
    cv::Mat a1(1, cols, CV_64FC1), a2(1, cols, CV_64FC1), a3(1, cols, CV_64FC1);
    cv::Mat delta_x(2, cols, CV_64FC1), xd(2, cols, CV_64FC1);

    // calculate distortion matrix
    for (i = 0; i < cols; i++) {
        r2.at<double>(0, i) =
            pow(x.at<double>(0, i), 2) + pow(x.at<double>(1, i), 2);
        r4.at<double>(0, i) = pow(r2.at<double>(0, i), 2);
        r6.at<double>(0, i) = pow(r2.at<double>(0, i), 3);
    }

    // calculate radical distortion matrix
    for (i = 0; i < cols; i++) {
        cdist.at<double>(0, i) = 1 + k.at<double>(0, 0) * r2.at<double>(0, i) +
                                 k.at<double>(1, 0) * r4.at<double>(0, i) +
                                 k.at<double>(2, 0) * r6.at<double>(0, i);
        xd1.at<double>(0, i) = x.at<double>(0, i) * cdist.at<double>(0, i);
        xd1.at<double>(1, i) = x.at<double>(1, i) * cdist.at<double>(0, i);
    }

    // calculate tangential distortion matrix
    for (i = 0; i < cols; i++) {
        a1.at<double>(0, i) = 2 * x.at<double>(0, i) * x.at <double>(1, i);
        a2.at<double>(0, i) =
            r2.at<double>(0, i) + 2 * pow(x.at<double>(0, i), 2);
        a3.at<double>(0, i) =
            r2.at<double>(0, i) + 2 * pow(x.at<double>(1, i), 2);
    }

    for (i = 0; i < cols; i++) {
        delta_x.at<double>(0, i) = k.at<double>(2, 0) * a1.at<double>(0, i) +
                                   k.at<double>(3, 0) * a2.at<double>(0, i);
        delta_x.at<double>(1, i) = k.at<double>(2, 0) * a3.at<double>(0, i) +
                                   k.at<double>(3, 0) * a1.at<double>(0, i);
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

cv::Mat normalize_pixel(cv::Mat x_kk, cv::Mat fc, cv::Mat cc, cv::Mat kc,
                        double alpha_c) {
    double row = x_kk.rows, col = x_kk.cols;
    cv::Mat xn(row, col, CV_64FC1), x_distort(row, col, CV_64FC1);

    cv::vconcat(
        (x_kk.rowRange(0, 1) - cc.at<double>(0, 0)) / fc.at<double>(0, 0),
        (x_kk.rowRange(1, 2) - cc.at<double>(1, 0)) / fc.at<double>(1, 0),
        x_distort);
    x_distort.rowRange(0, 1) =
        x_distort.rowRange(0, 1) - alpha_c * x_distort.rowRange(1, 2);
    xn = comp_distortion_oulu(x_distort, kc);
    return xn;
}

cv::Mat orig2rect(cv::Mat pix, cv::Mat intrMatOld, cv::Mat intrMatNew,
                  cv::Mat R, cv::Mat kc) {
    cv::Mat pixUndist, pixUndistR, pixRect;
    cv::Mat pix_transpose, pixUndistHomo;

    cv::transpose(pix, pix_transpose);
    cv::Mat input1 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 0),
                      intrMatOld.at<double>(1, 1));
    cv::Mat input2 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 2),
                      intrMatOld.at<double>(1, 2));
    pixUndist = normalize_pixel(pix_transpose, input1, input2, kc, 0);

    cv::Mat monesmat = cv::Mat::ones(1, pixUndist.cols, CV_64FC1);
    cv::vconcat(pixUndist, monesmat, pixUndistHomo);
    pixRect = intrMatNew * pixUndistR;

    cv::vconcat(pixRect.rowRange(0, 1) / pixRect.rowRange(2, 3),
                pixRect.rowRange(1, 2) / pixRect.rowRange(2, 3), pixRect);
    cv::transpose(pixRect, pixRect);
    return pixRect;
}

cv::Mat remap_rect(cv::Mat pixRect, cv::Mat KDistort, cv::Mat KRect, cv::Mat R,
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
                rays2.rowRange(1, 2) / rays2.rowRange(2, 3), x);

    xd = apply_distortion(x, distCoeff);

    px2 = KDistort.at<double>(0, 0) *
              (xd.rowRange(0, 1) + alpha * xd.rowRange(1, 2)) +
          KDistort.at<double>(0, 2);
    py2 = KDistort.at<double>(1, 1) * xd.rowRange(1, 2) +
          KDistort.at<double>(1, 2);

    cv::vconcat(px2, py2, pixDist);
    cv::transpose(pixDist, pixDist);
    return pixDist;
}

cv::Mat makeofst2(cv::Mat inValidY, cv::Mat deltaLut) {
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
                imValidYUp.at<double>(rows, ii) =
                    imValidYUp.at<double>(max, ii);
            }
        }
    }

    for (ii = 0; ii < imValidYDown.cols; ii++) {
        for (rows = 0; rows < imValidYDown.rows; rows++) {
            if (imValidYDown.at<double>(rows, ii) != 0) max = rows;
            break;
        }
    }
    for (rows = 0; rows < imValidYDown.rows; rows++) {
        if (imValidYDown.at<double>(rows, ii) != 0) {
            imValidYDown.at<double>(rows, ii) =
                imValidYDown.at<double>(max, ii);
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

lutVecName bilinear2x2(int coordtype, bool reverseMapping, cv::Mat xMatSampled,
                       cv::Mat yMatSampled, cv::Mat lut, initParams* params) {
    lutVecName lutVecNameN;
    std::string lutVecCharN;
    cv::Mat lutVecN;

    const int t = params->int_len + params->frac_len;
    double nr = params->img_size[0], nc = params->img_size[1];
    int row, col;
    for (row = 0; row < lut.rows; row++) {
        for (col = 0; col < lut.cols; col++) {
            lut.ptr<double>(row)[col] =
                cvRound(pow(2, 5) * lut.ptr<double>(row)[col]) /
                (double)pow(2, 5);
        }
    }

    cv::Mat lut2;
    if (coordtype == 1) {
        lut2 = xMatSampled - lut;
    } else {
        lut2 = yMatSampled - lut;
    }

    if (reverseMapping == false) {
        params->int_len = params->int_len - 1;
    }

    cv::Mat lutVec1, lutVec2, lutVec3;
    cv::transpose(xMatSampled.rowRange(0, 1), lutVec1);
    lutVec1 = lutVec1 * pow(2, t);
    lutVec2 = yMatSampled.colRange(0, 1) * pow(2, t - 1);

    for (col = 0; col < lut.cols; col++) {
        lutVec3.push_back(lut2.colRange(col, col + 1));
    }

    lutVec3 = lutVec3 * pow(2, params->int_len);

    lutVecN.push_back(nc);
    lutVecN.push_back(nr);
    lutVecN.push_back(params->upcrop[0]);
    lutVecN.push_back(params->upcrop[1]);
    lutVecN.push_back(params->downcrop[0]);
    lutVecN.push_back(params->downcrop[1]);
    lutVecN.push_back(lut.cols);
    lutVecN.push_back(lut.rows);
    lutVecN.push_back(lutVec1);
    lutVecN.push_back(lutVec2);
    lutVecN.push_back(lutVec3);

    size_t step1 = 8 + xMatSampled.cols;
    size_t step2 = step1 + yMatSampled.rows;
    size_t step3 = step2 + yMatSampled.rows * yMatSampled.cols;

    std::string bin_x;
    for (size_t i = 0; i < lutVecN.rows; i++) {
        if (lutVecN.ptr<double>(i)[0] >= 0) {
            if (i < 8) {
                bin_x = dec2bin(lutVecN.ptr<double>(i)[0], t - 1);
            } else if (i >= 8 && i < step1) {
                bin_x = dec2bin(lutVecN.ptr<double>(i)[0], t);
            } else if (i >= step1 && i < step2) {
                bin_x = dec2bin(lutVecN.ptr<double>(i)[0], t - 1);
            } else {
                bin_x = dec2bin(lutVecN.ptr<double>(i)[0], t);
            }
            lutVecCharN += bin_x;
        } else {
            if (i < 8) {
                bin_x =
                    dec2bin(pow(2, t - 1) + lutVecN.ptr<double>(i)[0], t - 1);
            } else if (i >= 8 && i < step1) {
                bin_x = dec2bin(pow(2, t) + lutVecN.ptr<double>(i)[0], t);
            } else if (i >= step1 && i < step2) {
                bin_x =
                    dec2bin(pow(2, t - 1) + lutVecN.ptr<double>(i)[0], t - 1);
            } else {
                bin_x = dec2bin(pow(2, t) + lutVecN.ptr<double>(i)[0], t);
            }
            lutVecCharN += bin_x;
        }
    }

    lutVecNameN.lut_vecchar = lutVecCharN;
    lutVecNameN.lut_vec = lutVecN;

    return lutVecNameN;
}

lutXY bilinear_interp(bool reverseMapping, cv::Mat xMatSampled,
                      cv::Mat yMatSampled, cv::Mat xOrig2Rect,
                      cv::Mat yOrig2Rect, initParams* params) {
    lutXY lutVecXY;

    lutVecXY.x = bilinear2x2(1, reverseMapping, xMatSampled, yMatSampled,
                             xOrig2Rect, params);
    lutVecXY.y = bilinear2x2(2, reverseMapping, xMatSampled, yMatSampled,
                             yOrig2Rect, params);

    return lutVecXY;
}

lutXY gen_lut(initParams* params, cv::Mat intrOld, cv::Mat kc, cv::Mat intrNew,
              cv::Mat rotMat, bool reverseMapping, char const* which_cam,
              cv::Mat sampleX, cv::Mat sampleY) {
    lutXY genLutXY;

    double nc = params->img_size[0], nr = params->img_size[1], thr;
    double lutSize[2] = {static_cast<double>(std::max(sampleY.rows, sampleY.cols)),
                         static_cast<double>(std::max(sampleX.rows, sampleX.cols))};
    cv::Mat xMatSampled, yMatSampled, yMatSampled_mid, pixSampled, pixSampledX,
        pixSampledY;
    cv::transpose(sampleY, yMatSampled_mid);
    cv::transpose(sampleY, yMatSampled);
    pixSampledX = xMatSampled.colRange(0, 1);
    pixSampledY = yMatSampled.colRange(0, 1);
    for (int i = 0; i < sampleY.cols; i++) {
        xMatSampled.push_back(sampleX);
    }
    for (int j = 0; j < sampleX.cols; j++) {
        cv::hconcat(yMatSampled, yMatSampled_mid, yMatSampled);
    }
    for (int i = 0; i < xMatSampled.cols; i++) {
        pixSampledX.push_back(xMatSampled.colRange(i, i + 1));
        pixSampledY.push_back(yMatSampled.colRange(i, i + 1));
        cv::hconcat(pixSampledX, pixSampledY, pixSampled);
    }

    cv::Mat pixOrigSampled;
    if (!reverseMapping) {
        pixOrigSampled = remap_rect(pixSampledX, intrOld, intrNew, rotMat, kc);
    } else {
        pixOrigSampled = orig2rect(pixSampledX, intrOld, intrNew, rotMat, kc);
    }

    cv::Mat xOrig = cv::Mat::zeros(lutSize[0], lutSize[1], CV_64F);
    cv::Mat yOrig = cv::Mat::zeros(lutSize[0], lutSize[1], CV_64F);
    for (size_t j = 0; j < lutSize[1]; j++) {
        for (size_t i = 0; i < lutSize[0]; i++) {
            xOrig.ptr<double>(i)[j] =
                pixOrigSampled.ptr<double>(j * lutSize[1] + i)[0];
            yOrig.ptr<double>(i)[j] =
                pixOrigSampled.ptr<double>(j * lutSize[0] + i)[1];
        }
    }

    cv::Mat deltaLutX2, deltaLutY2, deltaLutX, deltaLutY;
    cv::Mat inValidLutX = cv::Mat::zeros(lutSize[0], lutSize[1], CV_64F);
    cv::Mat inValidLutY = cv::Mat::zeros(lutSize[0], lutSize[1], CV_64F);
    cv::Mat validMat = cv::Mat::zeros(lutSize[0], lutSize[1], CV_64F);

    if (!reverseMapping) {
        deltaLutX2 = xMatSampled - xOrig;
        deltaLutY2 = yMatSampled - yOrig;
        thr = 128;
    } else {
        for (size_t i = 0; i < xOrig.rows; i++) {
            for (size_t j = 0; j < yOrig.rows; j++) {
                if (cvFloor(xOrig.ptr<double>(i)[j]) >= 1 &&
                    cvFloor(xOrig.ptr<double>(i)[j]) <= nc &&
                    cvFloor(yOrig.ptr<double>(i)[j]) >= 1 &&
                    cvFloor(yOrig.ptr<double>(i)[j]) <= nr) {
                    validMat.ptr<double>(i)[j] = 1;
                }
                if (cvCeil(xOrig.ptr<double>(i)[j]) >= 1 &&
                    cvCeil(xOrig.ptr<double>(i)[j]) <= nc &&
                    cvCeil(yOrig.ptr<double>(i)[j]) >= 1 &&
                    cvCeil(yOrig.ptr<double>(i)[j]) <= nr) {
                    validMat.ptr<double>(i)[j] = 1;
                }
            }
        }

        for (size_t j = 0; j < validMat.cols; j++) {
            validMat.ptr<double>(0)[j] = 0;
            validMat.ptr<double>(validMat.rows - 1)[j] = 0;
        }
        for (size_t i = 0; i < validMat.rows; i++) {
            validMat.ptr<double>(i)[0] = 0;
            validMat.ptr<double>(i)[validMat.cols - 1] = 0;
        }

        cv::Mat kernelstruct =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(validMat, validMat, kernelstruct);

        deltaLutX2 = xMatSampled - xOrig;
        deltaLutY2 = yMatSampled - yOrig;

        for (size_t i = 0; i < validMat.rows; i++) {
            for (size_t j = 0; j < validMat.cols; j++) {
                if (validMat.ptr<double>(i)[j] == 0) {
                    inValidLutX.ptr<double>(i)[j] = 0;
                    inValidLutY.ptr<double>(i)[j] = 0;
                } else {
                    inValidLutX.ptr<double>(i)[j] =
                        deltaLutX2.ptr<double>(i)[j];
                    inValidLutY.ptr<double>(i)[j] =
                        deltaLutY2.ptr<double>(i)[j];
                }
            }
        }

        deltaLutX2 = makeofst2(inValidLutX, deltaLutX);
        deltaLutY2 = makeofst2(inValidLutY, deltaLutY);

        thr = 256;
    }

    cv::Mat xOrig2Rect, yOrig2Rect, deltaDlutX2, deltaDlutY2;
    deltaDlutX2 = deltaLutX2;
    deltaDlutY2 = deltaLutY2;
    for (size_t i = 0; i < deltaDlutX2.rows; i++) {
        for (size_t j = 0; j < deltaDlutX2.cols; j++) {
            if (deltaDlutX2.ptr<double>(i)[j] > thr) {
                deltaDlutX2.ptr<double>(i)[j] = thr - 1;
            }
            if (deltaDlutX2.ptr<double>(i)[j] < -thr) {
                deltaDlutX2.ptr<double>(i)[j] = -thr + 1;
            }
            if (deltaDlutY2.ptr<double>(i)[j] > thr) {
                deltaDlutY2.ptr<double>(i)[j] = thr - 1;
            }
            if (deltaDlutY2.ptr<double>(i)[j] < -thr) {
                deltaDlutY2.ptr<double>(i)[j] = -thr + 1;
            }
        }
    }

    xOrig2Rect = xMatSampled - deltaDlutX2;
    yOrig2Rect = yMatSampled - deltaDlutY2;

    genLutXY = bilinear_interp(reverseMapping, xMatSampled, yMatSampled,
                               xOrig2Rect, yOrig2Rect, params);
    return genLutXY;
}

void write_camparam(char const* path, cv::Mat intrNewL, cv::Mat intrNewR,
                    cv::Mat transVec) {
    FILE* fp;

    double r_stereo_t_vec[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    char const name[] = "camparam.txt";
    char tr[300];
    strcpy(tr, path);
    strcpy(tr, name);

    fp = fopen(tr, "w");
    fprintf(fp, "%d %d %d\n", imgsize0[1], imgsize0[0], 0);
    fprintf(fp, "pinhole\n");
    fprintf(fp, "%d\n", 2);
    fprintf(fp, "%0.6f %0.6f %0.6f %0.6f\n", intrNewL.at<double>(0, 0),
            intrNewL.at<double>(1, 1), intrNewL.at<double>(0, 2),
            intrNewL.at<double>(1, 2));
    fprintf(fp, "%0.6f %0.6f %0.6f %0.6f\n", intrNewR.at<double>(0, 0),
            intrNewR.at<double>(1, 1), intrNewR.at<double>(0, 2),
            intrNewR.at<double>(1, 2));
    fprintf(fp,
            "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f "
            "%0.6f\n",
            r_stereo_t_vec[0], r_stereo_t_vec[1], r_stereo_t_vec[2],
            r_stereo_t_vec[3], r_stereo_t_vec[4], r_stereo_t_vec[5],
            r_stereo_t_vec[6], r_stereo_t_vec[7], r_stereo_t_vec[8],
            -cv::norm(transVec), 0.0f, 0.0f);
    fclose(fp);
}
