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
    cv::Mat y_sampled(y_size, 1, CV_64F);
    for (int i=0; i<y_size; i++) {
        y_sampled.at<double>(i, 0) = scale + yStep * i;
    }
    if (y_sampled.at<double>(y_sampled.size[0]-1, 0) != nr) {
        cv::Mat y_sample(y_size+1, 1, CV_64F);
        for (int i=0; i<y_size; i++) {
            y_sample.at<double>(i, 0) = y_sampled.at<double>(i, 0);
        }
        y_sample.at<double>(y_sample.size[0]-1, 0) = nr;
    }

    for (int ii=scale; ii<scale*(nc-1); ii+=scale) {
        int i = ii / scale;
        cv::Mat pixRect_1;
        if (useTable) {
            cv::Mat x_slice, y_slice;
            cv::transpose(xOrig2Rect(cv::Range(y_sampled.at<double>(i), y_sampled.at<double>(i)+scale-1), cv::Range::all()), x_slice);
            cv::transpose(yOrig2Rect(cv::Range(y_sampled.at<double>(i), y_sampled.at<double>(i)+scale-1), cv::Range::all()), y_slice);
            cv::hconcat(x_slice, y_slice, pixRect_1);
            }
        else {
            cv::Mat xuuAllUse_slice, yuuAllUse_slice, pix_tmp, pix;
            xuuAllUse_slice = xuuAllUse(cv::Range(y_sampled.at<double>(i), y_sampled.at<double>(i)), cv::Range::all());
            yuuAllUse_slice = yuuAllUse(cv::Range(y_sampled.at<double>(i), y_sampled.at<double>(i)), cv::Range::all());
            cv::vconcat(xuuAllUse_slice, yuuAllUse_slice, pix_tmp);
            cv::transpose(pix_tmp, pix);
            pixRect_1 = Orig2Rect(pix, Kinit, KK_new, R, K);
        }

        // ceiling operation for every pixRect_1 ???
        for (int i=0; i<pixRect_1.size[0]; i++) {
            for (int j=0; j<2; j++) {   // dimension is (N, 2)
                pixRect_1.at<double>(i, j) = ceil(pixRect_1.at<double>(i, j)); // this may not work sice ceil(double num) but not ceil(int num)
            }
        }

        // here is for cropping operation ... Not implemented
        // TODO...

        // ??? why would we do this?
        int row_pixRect_1, col_pixRect_1;
        row_pixRect_1 = pixRect_1.size[0];
        col_pixRect_1 = pixRect_1.size[1];
        if (row_pixRect_1 != 0 || col_pixRect_1 != 0) {
            if (int(pixRect_1.at<double>(0, 0)) % scale != 0) {
                pixRect_1.at<double>(0, 0) = pixRect_1.at<double>(0, 0) - 1;
            }
            else if (int(pixRect_1.at<double>(0, 1)) % scale != 0) {
                pixRect_1.at<double>(0 ,1) = pixRect_1.at<double>(0 ,1) - 1;
            }
        }

        if (row_pixRect_1 != 0 || col_pixRect_1 != 0) {
            cv::Mat startPixRect = pixRect_1.rowRange(0, 1).clone(); // [0, 1) left closed right open.
            cv::Mat pix = startPixRect;
            for (int j=startPixRect.at<double>(0, 0); j<row_pixRect_1; j++) {
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

            }
        }
    }

    cv::Mat pixAll_1;
    cv::hconcat(xuuAllUse, yuuAllUse, pixAll_1);
    
    return imgRect;
}