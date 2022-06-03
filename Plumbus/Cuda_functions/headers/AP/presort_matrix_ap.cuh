#pragma once
#include"../../../cuda_includes.h"
















void presort_matrix_ap_launch(cv::cuda::GpuMat& similarity_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& result, int N);