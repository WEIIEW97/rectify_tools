#include "lutParser.h"
#include "rectImg.h"
#include "utils.h"

int main() {
    std::string lut_file =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "output/LutDecL_640_400.txt";
    std::string input_image_file =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "imgL/IR-left-2022-4-8-18-28-20-654-1-996-57660643.bmp";
    std::string yuv_path =
        "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/"
        "output/rectL_996.yuv";
    const int int_len = 9;
    const int frac_len = 5;
    cv::Mat Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX,
        Rect2RawDenseMapY;
    lut_parser(lut_file, int_len, frac_len, Raw2RectDenseMapX,
               Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY);

    cv::Mat y, u, v;
    get_yuv(yuv_path, 640, 400, y, u, v);
    cv::Mat y_buffer[] = {y, y, y};
    cv::Mat y_ch;
    cv::merge(y_buffer, 3, y_ch);
    cv::Mat y_rect;
    y_rect = rect_img(Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX,
                      Rect2RawDenseMapY, y_ch, 1);
    cv::imshow("y_rect", y_rect);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}