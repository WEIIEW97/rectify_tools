#include "use_lut.h"

void use_lut(int row, int col, int int_len, int frac_len,
             const std::string& lut_file, const std::string& in_path,
             const std::string& out_path, bool show_bi_img, bool is_rgb) {
    cv::Mat xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig;
    lut_parser(lut_file, int_len, frac_len, xOrig2Rect, yOrig2Rect, xRect2Orig,
               yRect2Orig);

    cv::Mat src;
    unsigned char* yuv420_buf = readyuv(in_path, col, row, SFmt::YUV420);
    unsigned char* yuv444_buf = convert_yuv(yuv420_buf, col, row, CFmt::YUV420toYUV444);
    src = yuv2mat(yuv444_buf, col, row, SFmt::YUV444);

    delete[] yuv420_buf;
    delete[] yuv444_buf;
    
    rectBuffer y_rect;
    y_rect = rect_img(xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig, src);

    unsigned char* dst_buf = mat2yuv(y_rect.rect_img, SFmt::YUV444);
    unsigned char* dst420_buf = convert_yuv(dst_buf, col, row, CFmt::YUV444toYUV420);
    
    cv::Mat rect_y;
    cv::extractChannel(y_rect.rect_img, rect_y, 0);

    std::vector<int> _param = {0};
    cv::imwrite(out_path, rect_y, _param);
    std::string yuv_path = "D:/rectify_tools/cpp/data/output/rect1920.yuv";
    writeyuv(yuv_path, dst420_buf, col, row, SFmt::YUV420);
    delete[] dst_buf;
    delete[] dst420_buf;
    std::vector<int> unrect;
    unrect = get_unrect_idx(row, col, y_rect.rect_idx);

    if (show_bi_img) {
        cv::Mat bi_img;
        bi_img = show_bilinear_img(row, col, unrect);
        show_img(bi_img, BILINEAR_IMG_WINNAME);
    }

    printf("The number of unrectified pixels: %zu\n", unrect.size());
}
