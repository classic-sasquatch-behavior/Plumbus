#pragma once
#include"../../cuda_includes.h"






class CudaInterface {
public:
	CudaInterface();

	cv::Mat selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	cv::Mat fast_selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	std::vector<thrust::pair<int, int>> find_borders(cv::Mat labels);



#pragma region matrix operations

	void add(cv::Mat input_a, cv::Mat input_b, cv::Mat &output);
	void subtract(cv::Mat input_a, cv::Mat input_b, cv::Mat &output);
	void multiply(cv::Mat input_a, cv::Mat input_b, cv::Mat &output);

	template<typename AnyType>
	AnyType sum(cv::Mat input_a, cv::Mat input_b) {
		AnyType output;
		output = sum_launch<AnyType>(cv::Mat input_a, cv::Mat input_b);
		return output;
	}



#pragma endregion




private:

};