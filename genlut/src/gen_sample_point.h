// CamCalibTool - A camera calibration tool
// Copyright (c) 2022, Algorithm Development Team of NextVPU (Shanghai) Co., Ltd. All rights reserved.
//
// This software was developed of Jacob.lsx

#ifndef GEN_SAMPLE_POINT_H_
#define GEN_SAMPLE_POINT_H_

#include <opencv2/core/core.hpp>

void genSamplePoint(int img_w, int img_h,
                    cv::Mat& sampled_x1, cv::Mat& sampled_y1,
                    cv::Mat& sampled_x2, cv::Mat& sampled_y2);

#endif // !GEN_SAMPLE_POINT_H_

