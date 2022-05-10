#pragma once
#include"../../cuda_includes.h"





void selective_blur_launch(cv::cuda::GpuMat &d_src, cv::cuda::GpuMat &d_dst, int threshold, int kernel_min, int kernel_max, int width, int height);