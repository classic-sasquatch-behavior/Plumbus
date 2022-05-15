#pragma once
#include"../../cuda_includes.h"











void extract_exemplars_launch(cv::cuda::GpuMat &input, thrust::device_vector<int> &output, int N);