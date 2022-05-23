#pragma once
#include "../../cuda_includes.h"





void form_similarity_matrix_color_launch(cv::cuda::GpuMat &color_source, cv::cuda::GpuMat &coordinate_source, cv::cuda::GpuMat &output, int N);