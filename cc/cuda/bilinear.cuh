#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../src/utils.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_X_STEP (blockDim.x * gridDim.x)
#define CUDA_Y_STEP (blockDim.y * gridDim.y)
#define CUDA_Z_STEP (blockDim.z * gridDim.z)
// The following helper functions are here so that you can write a kernel call
// when you are not particularly interested in maxing out the kernels'
// performance. Usually, this will give you a reasonable speed, but if you
// really want to find the best performance, it is advised that you tune the
// size of the blocks and grids more reasonably.
// A legacy note: this is derived from the old good Caffe days, when I simply
// hard-coded the number of threads and wanted to keep backward compatibility
// for different computation capabilities.
// For more info on CUDA compute capabilities, visit the NVidia website at:
//    http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// 2D grid
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;

// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
// 1D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
// 2D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX = 128;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY = 128;

constexpr int kCUDAGridDimMaxX = 2147483647;
constexpr int kCUDAGridDimMaxY = 65535;
constexpr int kCUDAGridDimMaxZ = 65535;