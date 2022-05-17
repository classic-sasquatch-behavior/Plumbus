#pragma once
#include"../../cuda_includes.h"










void update_critereon_matrix_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& critereon_matrix, int N);













