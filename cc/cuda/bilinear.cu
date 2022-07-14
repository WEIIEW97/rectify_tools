#include "bilinear.cuh"

__host__ __device__ int32_t float2fixed_gpu(float num, int frac_len) {
  int32_t res;
  int _buffer;
  int int_part = 2 << (frac_len - 1);

  if (num > 0) {
    // after tons of experiments, should use flooring when input is positive
    // use ceiling when input is negative.
    _buffer = std::floor(num * int_part);
    res = (_buffer == int32_t(_buffer)) ? _buffer
                                        : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
  } else if (num < 0) {
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

__host__ __device__ float fixed2float_gpu(int32_t num, int frac_len) {
  int sign_flag = num & SIGN_BIT;  // 1 -> negative, 0 -> positive
  float res, _buffer;
  int32_t _buffer1;
  int int_part = 2 << (frac_len - 1);

  if (sign_flag == 0) {
    res = float(num) / int_part;
  } else {
    _buffer1 = num ^ SIGN_BIT;
    _buffer = float(_buffer1);
    res = _buffer / int_part;
    res = -res;
  }
  return res;
}

__host__ __device__ float num2fix_gpu(float num, int frac_len) {
  int32_t resf;
  float resd;
  resf = float2fixed_gpu(num, frac_len);
  resd = fixed2float_gpu(resf, frac_len);
  return resd;
}

void xy_coordinate(int row, int col, cv::Mat& XY) {
  cv::Mat x_all, y_all;
  meshgrid(cv::Range(1, col), cv::Range(1, row), x_all, y_all);

  x_all = x_all.t();
  y_all = y_all.t();

  cv::hconcat(x_all.reshape(0, row * col), y_all.reshape(0, row * col), XY);
}

__global__ void sparse2dense_gpu(float* y_sample,
                                 float* x_sample,
                                 float* xy_coordinate,
                                 float* sparse_,
                                 float* dense_,
                                 const int xrows,
                                 const int xcols,
                                 const int xyrows,
                                 const int y_sample_w,
                                 const int y_smaple_h,
                                 const int x_sample_w,
                                 const int x_sample_h,
                                 const int xy_sample_w,
                                 const int xy_sample_h,
                                 const int sparse_w,
                                 const int sparse_h,
                                 const int dense_w,
                                 const int dense_h) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t k = blockIdx.z * blockDim.z + threadIdx.z;

  for (i; i < xrows - 1; i += CUDA_X_STEP) {
    float* ptr_y = y_sample + i * y_sample_w;
    float* ptr_y_next = y_sample + (i + 1) * y_sample_w;
    float delta_y = *ptr_y_next - *ptr_y;
    float* ptr_x = x_sample;
    float* ptr_sparse = sparse_ + i * sparse_w;
    float* ptr_sparse_next = sparse_ + (i + 1) * sparse_w;
    for (j; j < xcols - 1; j += CUDA_Y_STEP) {
      float mark_x = ptr_x[j];
      float mark_x_next = ptr_x[j + 1];
      float delta_x = mark_x_next - mark_x;
      float delta_xy = delta_x * delta_y;
      for (k; k < xyrows; k += CUDA_Z_STEP) {
        /// TODO
      }
    }
  }
}