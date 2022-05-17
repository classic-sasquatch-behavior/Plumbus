#pragma once
#include"../../cuda_includes.h"






class CudaInterface {
public:
	CudaInterface();

	cv::Mat selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	//not fast lol
	cv::Mat fast_selective_blur(cv::Mat input, int steps, int threshold, int kernel_size);

	std::vector<thrust::pair<int, int>> find_borders(cv::Mat labels);
	
#pragma region affinity propagation

	void form_similarity_matrix(std::vector<cv::Mat> &input_histograms, cv::Mat &similarity_matrix, int N);
	void form_similarity_matrix_color(std::vector<float> &input_colors, cv::Mat &similarity_matrix, int N);
	void form_responsibility_matrix(cv::Mat &similarity_matrix, cv::Mat &responsibility_matrix, int N);
	void form_availibility_matrix(cv::Mat &responsibility_matrix, cv::Mat &availibility_matrix, int N);
	void form_critereon_matrix(cv::Mat &responsibility_matrix, cv::Mat &availibility_matrix, cv::Mat &critereon_matrix, int N);
	void extract_exemplars(cv::Mat &critereon_matrix, std::vector<int> &exemplars, int N);

	void affinity_propagation_color(std::vector<float> &input_colors, std::vector<int>& exemplars, int N);

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