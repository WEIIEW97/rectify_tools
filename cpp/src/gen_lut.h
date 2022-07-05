#ifndef GEN_LUT_H_
#define GEN_LUT_H_

#include "utils.h"

#define MAX_BIT 0xff

enum error_flag {
    nvp_success = 0,
    camlen_error = -1,
    camlrlen_error = -2,
    xmlpathname_error = -3,
    xmlroot_error = -4,
    imgsize_error = -5,
};

typedef struct optCell {
    cv::Mat cell_1;
    cv::Mat cell_2;
    cv::Mat cell_3;
    cv::Mat cell_4;
    cv::Mat cell_5;
    double cell_6;
    double cell_7;
} optCell;

typedef struct lutVecName {
    std::string lut_vecchar;
    cv::Mat lut_vec;
    // cv::Mat name_mat;
    // double name_num;
} lutVecName;

typedef struct lutXY {
    lutVecName x;
    lutVecName y;
} lutXY;

typedef struct lutLR {
    std::vector<unsigned int> l_data;
    std::vector<unsigned int> r_data;
} lutLR;

typedef struct initParams {
    double img_size[2];
    double upcrop[2];
    double downcrop[2];
    int int_len;
    int frac_len;
} initParams;

int imgsize0[2], imgsize[2], imgsize_half[2];
cv::Mat sampledX1, sampledY1, sampledX2, sampledY2;

std::vector<unsigned int> make_blendian_data(std::vector<unsigned int> number);

cv::Mat apply_distortion(cv::Mat x, cv::Mat k);

cv::Mat comp_distortion_oulu(cv::Mat x, cv::Mat k);

cv::Mat normalize_pixel(cv::Mat x_kk, cv::Mat fc, cv::Mat cc, cv::Mat kc,
                        double alpha_c);

cv::Mat orig2rect(cv::Mat pix, cv::Mat intrMatOld, cv::Mat intrMatNew,
                  cv::Mat R, cv::Mat kc);

cv::Mat remap_rect(cv::Mat pixRect, cv::Mat KDistort, cv::Mat KRect, cv::Mat R,
                   cv::Mat distCoeff);

cv::Mat makeofst2(cv::Mat inValidY, cv::Mat deltaLut);

lutVecName bilinear2x2(int coordtype, bool reverseMapping, cv::Mat xMatSampled,
                       cv::Mat yMatSampled, cv::Mat lut, initParams* params);

lutXY bilinear_interp(bool reverseMapping, cv::Mat xMatSampled,
                      cv::Mat yMatSampled, cv::Mat xOrig2Rect,
                      cv::Mat yOrig2Rect, initParams* params);

lutXY gen_lut(initParams* params, cv::Mat intrOld, cv::Mat kc, cv::Mat intrNew,
              cv::Mat rotMat, bool reverseMapping, char const* which_cam,
              cv::Mat sampleX, cv::Mat sampleY);

void get_cam_param(cv::Mat transVecT, cv::Mat intrMatOldLT,
                   cv::Mat intrMatOldRT, cv::Mat kcLT, cv::Mat kcRT,
                   cv::Mat intrMatNewLT, cv::Mat intrMatNewRT, cv::Mat rotMatLT,
                   cv::Mat rotMatRT);

std::vector<unsigned int> make_blendian_data(std::vector<unsigned int>num);

int cam_writememory(cv::Mat imgSizeT, void* CamAddr, int CamLen, void* CamLAddr,
                    void* CamRAddr, int CamLRLen);

int cam_writefile(cv::Mat imgSizeT, char const* path, bool is_mono = false,
                  char const* name_caml = "L", char const* name_camr = "R");

int xmlmemory(cv::Mat imgSizeT, char const* XmlPathName, void* CamAddr,
              int CamLen, void* CamLAddr, void* CamRAddr, int CamLRLen);

int xmlfile(cv::Mat imgSizeT, char const* XmlPathName, char const* path);

#endif  // GEN_LUT_H_
