#include "utils.h"


int main() {
    int scale=1;
    int yStep = scale;
    int nr = 6;
    int y_size = (nr-scale)/scale + 1;
    cv::Mat y_sampled(y_size, 1, CV_8UC1);
    for (int i = 0; i < y_size; i++) 
    {
        // for (int j = 0; j < 1; j++) 
        // {
            y_sampled.at<int>(i, 0) = scale + yStep * i;
            std::cout << scale + yStep * i << std::endl;
        // }
    }
    std::cout << y_sampled << std::endl;
    // std::cout << y_sampled.at<int>(y_sampled.size[0], y_sampled.size[1]) << std::endl;
    // std::cout << 'size ' << y_sampled.size() << std::endl;
    std::cout << y_sampled.at<int>(y_sampled.size[0]-1, 0) << std::endl;
    return 0;
}