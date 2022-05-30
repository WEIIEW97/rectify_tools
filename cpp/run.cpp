#include "src/use_lut.h"

int main() {
    const int row = 400;
    const int col = 640;
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

    use_lut(row, col, int_len, frac_len, lut_file, yuv_path, output_image_file,
            false);
    return 0;
}