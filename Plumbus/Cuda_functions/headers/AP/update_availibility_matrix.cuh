#pragma once
#include"../../../cuda_includes.h"






void update_availibility_matrix_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibity_matrix, int N);





