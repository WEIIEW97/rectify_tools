#include "utils.h"
#include "lutParser.h"

int main() {
    std::string lut_file = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/output/LutDecL_640_400.txt";
    std::string input_image_file = "/Users/williamwei/Codes/rectify_tools/rectify_tools/cpp/data/case3/imgL/IR-left-2022-4-8-18-28-20-654-1-996-57660643.bmp";
    int int_len = 9;
    int frac_len = 5;
    cv::Mat Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY;
    lutParser(lut_file, input_image_file, int_len, frac_len, Raw2RectDenseMapX, Raw2RectDenseMapY, Rect2RawDenseMapX, Rect2RawDenseMapY);
//    std::cout << "Raw2RectDenseMapX: " << Raw2RectDenseMapX << std::endl;
//    std::cout << Raw2RectDenseMapX.size() << std::endl;
    return 0;
}