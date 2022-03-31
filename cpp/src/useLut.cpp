#include "utils.h"


int main() {
    cv::Mat ans;
    uchar arr[10] = {17, 24, 23, 5, 4, 6, 10, 12, 11, 18};
    cv::Mat pixTmp(5,2,CV_8UC1,arr);
    std::cout << pixTmp << std::endl;
    uchar arr1[] = {1,1,1,1,1};
    cv::Mat flagIn(5,1,CV_8UC1,arr1);

    return 0;
}