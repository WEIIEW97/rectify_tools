#include "lut_parser.h"
#include "rect_img.h"
#include "utils.h"

int main() {
    const std::string lut_file =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "output/LutDecL_640_400.txt";
    const std::string yuv_path =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "output/rectL_996.yuv";
    const std::string output_image_file =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "output/rectL_cpp1.png";
    const int int_len = 9;
    const int frac_len = 5;
    // clock_t start, end;
    // start = clock();
    cv::Mat xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig;
    lut_parser(lut_file, int_len, frac_len, xOrig2Rect, yOrig2Rect, xRect2Orig,
               yRect2Orig);
    //    end = clock();
    //    std::cout << "Opencv version: " << (double)(end - start) /
    //    CLOCKS_PER_SEC
    //              << " seconds" << std::endl;
    cv::Mat y, u, v;
    get_yuv(yuv_path, 640, 400, y, u, v);
    cv::Mat y_buffer[] = {y, y, y};
    cv::Mat y_ch;
    cv::merge(y_buffer, 3, y_ch);
    cv::Mat y_rect;
    y_rect = rect_img(xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig, y_ch, 1);
    // end = clock();
    // std::cout << "Runtime: " << double(end - start) / CLOCKS_PER_SEC << "s"
    //           << std::endl;
    cv::imshow("y_rect", y_rect);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // std::vector<int> compression_params = {0};
    // cv::imwrite(output_image_file, y_rect, compression_params);
    return 0;
}