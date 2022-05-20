#pragma once
#include"../../cuda_includes.h"











void extract_exemplars_launch(cv::cuda::GpuMat &input, cv::cuda::GpuMat &output, int N);