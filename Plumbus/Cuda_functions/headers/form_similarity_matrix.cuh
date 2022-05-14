#pragma once
#include "../../cuda_includes.h"





void form_similarity_matrix_launch(cv::cuda::GpuMat &source, cv::cuda::GpuMat &output, int N);