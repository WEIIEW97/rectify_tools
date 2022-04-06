#include "utils.h"


int main() {
//     cv::Mat ans;
//     double data[10] = {17, 24, 23, 5, 4, 6, 10, 12, 11, 18};
//     cv::Mat pixTmp = cv::Mat(5, 2, CV_64F, data);
//     std::cout << pixTmp << std::endl;
//     cv::Mat flagIn(5,1,CV_64F);
//     int nc=400, nr=640;
//     for (int i=0; i<5; i++) {
//         if (pixTmp.at<double>(i, 0) > 0 && pixTmp.at<double>(i, 0) <= nc && pixTmp.at<double>(i, 1) > 0 && pixTmp.at<double>(i, 1) <= nr) {
//             flagIn.at<double>(i) = 1;
//         }
//         else {
//             flagIn.at<double>(i) = 0;
//         }
//         // std::cout << int(pixTmp.at<double>(i, 0)) << std::endl;
//         // std::cout << int(pixTmp.at<double>(i, 1)) << std::endl;
//     }
//     cv::Mat flagin;
//     cv::bitwise_and();
//     std::cout << flagIn << std::endl;
//     return 0;
    double data[9] = {17, 24, 23, 5, 4, 6, 10, 12, 11};
    cv::Mat a = cv::Mat(3, 3, CV_64F, data);
    double data1[9] = {1, 5, 22, 11, 12, 7, 21, 65, 44};
    cv::Mat b = cv::Mat(3, 3, CV_64F, data1);

    cv::Mat c = a - b;

    cv::Mat d;
    cv::subtract(a, b, d);

    std::cout << c << std::endl;
    std::cout << d << std::endl;
}