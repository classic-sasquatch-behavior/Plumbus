#include"../../cuda_includes.h"
#include"cuda_function_includes.h"
#include"../../classes.h"
#include"../../config.h"

#pragma region constructors

CudaInterface::CudaInterface() {

}

#pragma endregion

#pragma region misc. GPU functions

std::vector<thrust::pair<int, int>> CudaInterface::find_borders(cv::Mat labels) {
	std::vector<thrust::pair<int, int>> h_out = find_borders_launch(labels.cols, labels.rows, labels);
	return h_out;
}

cv::Mat CudaInterface::selective_blur(cv::Mat input, int steps, int threshold, int kernel_size) {
	cv::Mat output(input.size(), input.type());
	int width = input.cols;
	int height = input.rows;
	int kernel_max = (kernel_size - 1) / 2;
	int kernel_min = -1*kernel_max;

	cv::cuda::GpuMat d_src;
	d_src.upload(input);
	cv::cuda::GpuMat d_dst = d_src;

	for (int step = 0; step < steps; step++) {

		selective_blur_launch(d_src, d_dst, threshold, kernel_min, kernel_max, width, height);

		d_src = d_dst;
	}

	d_src.download(output);

	return output;
}

cv::Mat CudaInterface::fast_selective_blur(cv::Mat input, int steps, int threshold, int kernel_size) {
	if (kernel_size > 17) {
		std::cout << "ERROR: cannot run fast selective blur - kernel too large (limit 17)" << std::endl;
		return input;
	}

	cv::Mat output(input.size(), input.type());
	int width = input.cols;
	int height = input.rows;

	cv::cuda::GpuMat d_src;
	d_src.upload(input);
	cv::cuda::GpuMat d_dst = d_src;

	for (int step = 0; step < steps; step++) {

		fast_selective_blur_launch(d_src, d_dst, threshold, kernel_size, width, height);

		d_src = d_dst;
	}

	d_src.download(output);

	return output;
}

#pragma endregion



#pragma region affinity propagation

void CudaInterface::form_similarity_matrix(std::vector<cv::Mat>& input_histograms, cv::Mat& output_similarity_matrix, int N) {

	std::cout << " concatenating histograms..." << std::endl;
	cv::Mat concatenated_histograms(cv::Size(N * 3, 256), input_histograms[0].type());
	cv::Mat* mat_array = input_histograms.data();
	cv::hconcat(mat_array, input_histograms.size(), concatenated_histograms);

	cv::cuda::GpuMat source(concatenated_histograms.size(), concatenated_histograms.type());
	cv::cuda::GpuMat output(output_similarity_matrix.size(), output_similarity_matrix.type());

	source.upload(concatenated_histograms);
	output.upload(output_similarity_matrix);

	std::cout << "launching similarity matrix..." << std::endl;
	form_similarity_matrix_launch(source, output, N);

	output.download(output_similarity_matrix);

	//cudaDeviceReset();
}

void CudaInterface::form_responsibility_matrix(cv::Mat& similarity_matrix, cv::Mat& responsibility_matrix, int N) {

}

void CudaInterface::form_availibility_matrix(cv::Mat& responsibility_matrix, cv::Mat& availibility_matrix, int N) {

}

void CudaInterface::form_critereon_matrix(cv::Mat& responsibility_matrix, cv::Mat& availibility_matrix, cv::Mat &critereon_matrix, int N) {

}


#pragma endregion



#pragma region matrix operations

void CudaInterface::add(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	add_launch(input_a, input_b, output);
}

void CudaInterface::subtract(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	subtract_launch(input_a, input_b, output);
}

void CudaInterface::multiply(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	multiply_launch(input_a, input_b, output);
}









#pragma endregion