#ifndef PATCH_IM2COL_CUH_
#define PATCH_IM2COL_CUH_

#include <torch/extension.h>

#include "cuda_helper.h"
#include "nn/common/patch_im2col.h"

namespace patch_conv {
namespace nn {
namespace cuda {

template <typename T>
__global__ void PatchIm2Col2DKernel(
    const int64_t num_kernels, const T *data_im_ptr, const int64_t patches,
    const int64_t height_im, const int64_t width_im, const int64_t height_out,
    const int64_t width_out, const int64_t width_col, const int64_t kernel_h,
    const int64_t kernel_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t dilation_h,
    const int64_t dilation_w, T *data_col_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::PatchIm2Col2D(index, data_im_ptr, patches, height_im, width_im,
                        height_out, width_out, width_col, kernel_h, kernel_w,
                        pad_h, pad_w, stride_h, stride_w, dilation_h,
                        dilation_w, data_col_ptr);
}

void PatchIm2Col2DLauncher(torch::Tensor data_im, const int64_t channels,
                           const int64_t patches, const int64_t height_im,
                           const int64_t width_im, const int64_t height_out,
                           const int64_t width_out, const int64_t width_col,
                           const int64_t kernel_h, const int64_t kernel_w,
                           const int64_t pad_h, const int64_t pad_w,
                           const int64_t stride_h, const int64_t stride_w,
                           const int64_t dilation_h, const int64_t dilation_w,
                           torch::Tensor data_col) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch channels * width_col kernels, with each kernel responsible for
  // copying a the convolutions over a single channel.
  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "PatchIm2Col2DKernel", ([&] {
        PatchIm2Col2DKernel<<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), patches, height_im,
            width_im, height_out, width_out, width_col, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            data_col.data<scalar_t>());
        CUDA_CHECK(cudaGetLastError())
      }));
}

template <typename T>
__global__ void PatchCol2Im2DKernel(
    const int64_t num_kernels, const T *data_col_ptr, const int64_t patches,
    const int64_t height, const int64_t width, const int64_t output_height,
    const int64_t output_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t dilation_h, const int64_t dilation_w,
    T *data_im_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }
  common::PatchCol2Im2D(index, data_col_ptr, patches, height, width,
                        output_height, output_width, kernel_h, kernel_w, pad_h,
                        pad_w, stride_h, stride_w, dilation_h, dilation_w,
                        data_im_ptr);
}

void PatchCol2Im2DLauncher(torch::Tensor data_col, const int64_t channels,
                           const int64_t patches, const int64_t height,
                           const int64_t width, const int64_t output_height,
                           const int64_t output_width, const int64_t kernel_h,
                           const int64_t kernel_w, const int64_t pad_h,
                           const int64_t pad_w, const int64_t stride_h,
                           const int64_t stride_w, const int64_t dilation_h,
                           const int64_t dilation_w, torch::Tensor data_im) {
  const int64_t num_kernels = channels * patches * height * width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "PatchCol2Im2DKernel", ([&] {
        PatchCol2Im2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(), patches, height, width,
            output_height, output_width, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            (data_im.data<scalar_t>()));
      }));
  CUDA_CHECK(cudaGetLastError())
}

}  // namespace cuda
}  // namespace nn
}  // namespace patch_conv
#endif