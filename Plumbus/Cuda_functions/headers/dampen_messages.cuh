#pragma once
#include"../../cuda_includes.h"









void dampen_messages_launch(cv::cuda::GpuMat& previous_matrix, cv::cuda::GpuMat& updated_matrix, float damping_factor, int N);