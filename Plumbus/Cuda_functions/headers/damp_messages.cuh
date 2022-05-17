#pragma once
#include"../../cuda_includes.h"









void damp_messages_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& previous_responsibility_matrix, cv::cuda::GpuMat& previous_availibility_matrix, float damping_factor, int N);