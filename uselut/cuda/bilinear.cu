#include "bilinear.h"

__device__ __forceinline__ int32_t double2fixed(double num, int frac_len) {
    int32_t res;
    int _buffer;
    int int_part = 2 << (frac_len - 1);
    if (res > 0) {
        _buffer = std::floor(num * int_part);
        res = (_buffer == int32_t(_buffer)) ? _buffer
                                            : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
    } else if (res < 0) {
        num = -num;
        _buffer = std::ceil(num * int_part);
        res = (_buffer == int32_t(_buffer)) ? _buffer
                                            : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
        res = res ^ SIGN_BIT;
    } else {
        res = 0;
    }
    return res;
}

__device__ __forceinline__ double fixed2double(int32_t num, int frac_len) {
    int sign_flag = num & SIGN_BIT;
    double res, _buffer;
    int32_t _buffer1;
    int int_part = 2 << (frac_len - 1);

    if (sign_flag == 0) {
        res = double(num) / int_part;
    } else {
        _buffer1 = num ^ SIGN_BIT;
        _buffer = double(_buffer1);
        res = _buffer / int_part;
        res = -res;
    }
    return res;
}

__device__ __forceinline__ double num2fix(double num, int frac_len) {
    int32_t resf;
    double resd;
    resf = double2fixed(num, frac_len);
    resd = fixed2double(resf, frac_len);
    return resd;
}

__device__ __forceinline__ double num2fix_unsigned(double num, int frac_len) {
    int32_t resf;
    double resd, res;
    resf = double2fixed(num, frac_len);
    resd = fixed2double(resf, frac_len);
    res = (resd > 0) ? resd : 0;
    return res;
}



