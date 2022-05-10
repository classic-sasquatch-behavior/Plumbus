#include"../../cuda_includes.h"
#include"../../classes.h"
#include"../headers/selective_blur.h"
#include"../headers/fast_selective_blur.cuh"
#include"../headers/find_borders.cuh"
#include"../../config.h"


#pragma region constructors

CudaInterface::CudaInterface() {

}

#pragma endregion

#pragma region GPU functions

std::set<int[2]> CudaInterface::find_borders(cv::Mat labels, int num_superpixels) {
	
	int num_field_rows = labels.rows;
	int num_field_cols = labels.cols;
	const int num_potential_borders_per_pixel = 8;
	const int num_integers_per_relationship = 2;

	int array_size = num_field_rows * num_field_cols * num_potential_borders_per_pixel * num_integers_per_relationship;















	//std::vector<int[2]> pairs;
	//std::set<int[2]> output;

	//pairs = find_borders_launch();

	//for (int[2] pair : pairs) {
	//	output.insert({pair[0], pair[1]});
	//	output.insert({pair[1], pair[0]});
	//}

	//return output;
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

