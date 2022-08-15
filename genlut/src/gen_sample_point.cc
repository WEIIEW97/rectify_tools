// CamCalibTool - A camera calibration tool
// Copyright (c) 2022, Algorithm Development Team of NextVPU (Shanghai) Co., Ltd. All rights reserved.
//
// This software was developed of Jacob.lsx

#include "gen_sample_point.h"

#include <iostream>
#include <vector>
#include <set>
#include <deque>
#include <numeric>

static cv::Mat genSamplePoint(int len, int thr);

void genSamplePoint(int img_w, int img_h,
                    cv::Mat& sampled_x1, cv::Mat& sampled_y1,
                    cv::Mat& sampled_x2, cv::Mat& sampled_y2) {
    sampled_x2 = genSamplePoint(img_w, 46);
    sampled_y2 = genSamplePoint(img_h, 36);
    sampled_x1 = sampled_x2.colRange(0, sampled_x2.size().width > 24 ? 24 : sampled_x2.size().width).clone();
    sampled_y1 = sampled_y2.clone();
}

cv::Mat genSamplePoint(int len, int thr) {
    int sample = std::ceil(float(len + 1) / 16);
    if (sample % 2 != 0)
        sample++;

    int half_sample = sample / 2 + std::floor(sample / thr);
    std::vector<int> scale(half_sample, 1);
    if (half_sample > thr / 2) {
        int index = 0;
        const std::set<int> limit = {1, 2, 4, 8};
        while (scale.size() > thr / 2) {
            scale.at(index)++;
            scale.resize(scale.size() - 1);
            while (!limit.count(scale.at(index))) {
                scale.at(index)++;
                if (scale.size() > thr / 2)
                    scale.resize(scale.size() - 1);
            }
            index++;
            if (index >= thr / 2)
                index = 0;
        }
    }

    std::deque<double> sample_point;
    for (int i = 0; i < scale.size(); i++) {
        double right = (len/2. + 0.5) - scale.at(0)*8. + std::accumulate(scale.begin(), scale.begin() + i + 1, 0)*16.;
        double left  = (len/2. + 0.5) + scale.at(0)*8. - std::accumulate(scale.begin(), scale.begin() + i + 1, 0)*16.;
        sample_point.emplace_front(left);
        sample_point.emplace_back(right);
    }

    cv::Mat sp(1, sample_point.size(), CV_64F);
    int i = 0;
    std::for_each(sample_point.cbegin(), sample_point.cend(), [&](const double value) {
        sp.at<double>(0, i++) = value; });
    return sp;
}