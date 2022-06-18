#include "use_lut.h"

void use_lut(int row, int col, int int_len, int frac_len,
             const std::string& lut_file, const std::string& yuv_path,
             const std::string& out_path, bool show_bi_img) {
    cv::Mat xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig;
    lut_parser(lut_file, int_len, frac_len, xOrig2Rect, yOrig2Rect, xRect2Orig,
               yRect2Orig);
    cv::Mat y, u, v;
    get_yuv(yuv_path, col, row, y, u, v);
    cv::Mat y_buffer[] = {y, y, y};
    cv::Mat y_ch;
    cv::merge(y_buffer, 3, y_ch);
    rectBuffer y_rect;
    y_rect = rect_img(xOrig2Rect, yOrig2Rect, xRect2Orig, yRect2Orig, y_ch, 1);
    
    std::vector<int> unrect;
    unrect = get_unrect_idx(row, col, y_rect.rect_idx);
    
    std::vector<int> _param = {0};
    cv::imwrite(out_path, y_rect.rect_img, _param);
    if (show_bi_img) {
        show_bilinear_img(row, col, unrect, BILINEAR_IMG_WINNAME);
    }

    
    printf("The number of unrectified pixels: %lu\n", unrect.size());
}
