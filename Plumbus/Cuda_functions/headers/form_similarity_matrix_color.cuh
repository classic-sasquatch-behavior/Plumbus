#pragma once
#include "../../cuda_includes.h"





void form_similarity_matrix_color_launch(thrust::device_vector<int> &source, cv::cuda::GpuMat &output, int N);