#pragma once
#include"../../../cuda_includes.h"





void fast_selective_blur_launch(cv::cuda::GpuMat& d_src, cv::cuda::GpuMat& d_dst, int threshold, int kernel_size, int width, int height);