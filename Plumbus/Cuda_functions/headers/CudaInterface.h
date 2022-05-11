#pragma once
#include"../../cuda_includes.h"






class CudaInterface {
public:
	CudaInterface();

	cv::Mat selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	cv::Mat fast_selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	std::vector<thrust::pair<int, int>> find_borders(cv::Mat labels);

private:

};