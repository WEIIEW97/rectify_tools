#include <rectimg.h>

// parse from outside
cv::Mat transVec(3, 1, CV_64FC1);
cv::Mat intrMatOldL(3, 3, CV_64FC1);
cv::Mat intrMatOldR(3, 3, CV_64FC1);
cv::Mat kcL(5, 1, CV_64FC1);
cv::Mat kcR(5, 1, CV_64FC1);
cv::Mat intrMatNewL(3, 3, CV_64FC1);
cv::Mat intrMatNewR(3, 3, CV_64FC1);
cv::Mat rotMatL(3, 3, CV_64FC1);
cv::Mat rotMatR(3, 3, CV_64FC1);

cv::Mat RectImg(cv::Mat xOrig2Rect, cv::Mat yOrig2Rect, cv::Mat xRect2Orig, cv::Mat yRect2Orig, cv::Mat image, cv::Mat KK_new,
cv::Mat Kinit, cv::Mat K, cv::Mat R, cv::Mat imgOrig, std::string paraDir, std::string whichOne, int scale) {
    cv::Mat imgRect = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    
    int useTable = 1;
    int draw = 0;
    int useNearest = 0;

    int nr = scale * image.rows;
    int nc = scale * image.cols;

    cv::Mat xuuAllUse, yuuAllUse;
    meshgrid(cv::Range(1, nc), cv::Range(1, nr), xuuAllUse, yuuAllUse);
    int yStep = scale;
    int y_size = (nr-scale)/scale + 1;
    cv::Mat y_sampled(y_size, 1, CV_8UC1);
    for (int i=0; i<y_size; i++) {
        y_sampled.at<int>(i, 0) = scale + yStep * i;
    }
    if (y_sampled.at<int>(y_sampled.size[0]-1, 0) != nr) {
        cv::Mat y_sample(y_size+1, 1, CV_8UC1);
        for (int i=0; i<y_size; i++) {
            y_sample.at<int>(i, 0) = y_sampled.at<int>(i, 0);
        }
        y_sample.at<int>(y_sample.size[0]-1, 0) = nr;
    }

    for (int ii=scale; ii<scale*(nc-1); ii+=scale) {
        int i = ii / scale;
        cv::Mat pixRect_1;
        if (useTable) {
            cv::Mat x_slice, y_slice;
            cv::transpose(xOrig2Rect(cv::Range(y_sampled.at<int>(i), y_sampled.at<int>(i)+scale-1), cv::Range::all()), x_slice);
            cv::transpose(yOrig2Rect(cv::Range(y_sampled.at<int>(i), y_sampled.at<int>(i)+scale-1), cv::Range::all()), y_slice);
            cv::hconcat(x_slice, y_slice, pixRect_1);
            }
        else {
            cv::Mat xuuAllUse_slice, yuuAllUse_slice, pix_tmp, pix;
            xuuAllUse_slice = xuuAllUse(cv::Range(y_sampled.at<int>(i), y_sampled.at<int>(i)), cv::Range::all());
            yuuAllUse_slice = yuuAllUse(cv::Range(y_sampled.at<int>(i), y_sampled.at<int>(i)), cv::Range::all());
            cv::vconcat(xuuAllUse_slice, yuuAllUse_slice, pix_tmp);
            cv::transpose(pix_tmp, pix);
            pixRect_1 = Orig2Rect(pix, Kinit, KK_new, R, K);
        }
    }
}