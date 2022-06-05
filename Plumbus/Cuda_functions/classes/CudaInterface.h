#pragma once
#include"../../cuda_includes.h"






class CudaInterface {
public:
	CudaInterface();

	cv::Mat selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	//not fast lol
	cv::Mat fast_selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	std::vector<thrust::pair<int, int>> find_borders(cv::Mat labels);


#pragma region SLIC
	cv::Mat SLIC_superpixels(cv::Mat& input, int density, int* num_superpixels);
	
	void enforce_connectivity(gMat &labels, int* num_superpixels, int threshold );

#pragma endregion



#pragma region affinity propagation



	void affinity_propagation_color(cv::Mat &colors, cv::Mat &coordinates, cv::Mat &exemplars, int N);

#pragma endregion






#pragma region matrix operations

	void add(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);
	void subtract(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);
	void multiply(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);

	//template<typename AnyType>
	//AnyType sum(cv::Mat &input_a, cv::Mat &input_b, AnyType &output) {
	//	sum_launch<AnyType>(input_a, input_b, output);
	//}



#pragma endregion




private:

};