#pragma once
#include"../../cuda_includes.h"









void update_responsibility_matrix_launch(cv::cuda::GpuMat& similarity_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& responsibility_matrix, int N);

















