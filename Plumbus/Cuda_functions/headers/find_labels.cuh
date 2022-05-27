#include"../../cuda_includes.h"







void find_labels_launch(cv::cuda::GpuMat& src_L, cv::cuda::GpuMat& src_A, cv::cuda::GpuMat& src_B, cv::cuda::GpuMat &labels, cv::cuda::GpuMat& row_vals, cv::cuda::GpuMat& col_vals, cv::cuda::GpuMat& sector_LUT, int density, int k_step);