#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>

#include "../src/utils.h"

__device__ __forceinline__ int32_t double2fixed(double num, int frac_len);

__device__ __forceinline__ double fixed2double(int32_t num, int frac_len);

__device__ __forceinline__ double num2fix(double num, int frac_len);

__device__ __forceinline__ double num2fix_unsigned(double num, int frac_len);

