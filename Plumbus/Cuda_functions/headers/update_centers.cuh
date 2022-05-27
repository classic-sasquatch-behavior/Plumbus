#include"../../cuda_includes.h"






void update_centers_launch(cv::cuda::GpuMat& labels, cv::cuda::GpuMat& row_vals, cv::cuda::GpuMat& col_vals, int* average_displacement);