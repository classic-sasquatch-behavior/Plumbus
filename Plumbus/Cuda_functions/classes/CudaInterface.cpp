#include"../../cuda_includes.h"
#include"cuda_function_includes.h"
#include"../../classes.h"
#include"../../config.h"



#pragma region structors

CudaInterface::CudaInterface() {
	if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
		cv::cuda::setDevice(0);
	}
	else {
		std::cout << "no cuda enabled device found, or driver not installed or something. [error from: CudaInterface.cpp]";
	}
	std::cout << std::endl;
	cv::cuda::printCudaDeviceInfo(0);
	std::cout << std::endl;
}

#pragma endregion

#pragma region misc GPU functions

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


void CudaInterface::affinity_propagation_color(std::vector<float>& input_colors, std::vector<int>& exemplars, int N) {

	std::cout << "running affinity propagation..." << std::endl;

	cv::Size matrix_size(N, N);
	float damping_factor = 0.5f;
	int convergence_threshold = 10;
	int matrix_type = CV_32FC1;
	bool algorithm_converged = false;

	cv::cuda::GpuMat similarity_matrix(matrix_size, matrix_type, cv::Scalar{0});
	cv::cuda::GpuMat responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat critereon_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });

	cv::cuda::GpuMat previous_responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat previous_availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });

	thrust::device_vector<float> source = input_colors;

	thrust::device_vector<int> working_exemplars = exemplars;

	std::vector<int> host_working_exemplars(N); 
	std::vector<int> exemplar_details(N);
	std::vector<int> previous_exemplar_details(N);
	
	std::cout << "forming similarity matrix..." << std::endl;

	form_similarity_matrix_color_launch(source, similarity_matrix, N);
	cv::Mat h_sim_mat(matrix_size, matrix_type);
	similarity_matrix.download(h_sim_mat);
	cv::Mat similarity_matrix_diagonal = h_sim_mat.diag(0);
	double lowest_val;
	cv::minMaxIdx(h_sim_mat, &lowest_val);
	similarity_matrix_diagonal.setTo((float)lowest_val);
	similarity_matrix.upload(h_sim_mat);
	util->print_gpu_mat(similarity_matrix, 5);

	while (!algorithm_converged) {
		std::cout << "updating responsibility matrix..." << std::endl;
		update_responsibility_matrix_launch(similarity_matrix, availibility_matrix, responsibility_matrix, N);
		util->print_gpu_mat(responsibility_matrix, 5);

		std::cout << "updating availibility matrix..." << std::endl;
		update_availibility_matrix_launch(responsibility_matrix, availibility_matrix, N);
		util->print_gpu_mat(availibility_matrix, 5);

		std::cout << "updating critereon matrix..." << std::endl;
		update_critereon_matrix_launch(responsibility_matrix, availibility_matrix, critereon_matrix, N);
		util->print_gpu_mat(critereon_matrix, 5);

		std::cout << "extracting exemplars..." << std::endl;

		extract_exemplars_launch(critereon_matrix, working_exemplars, N);
		
		thrust::copy(working_exemplars.begin(), working_exemplars.end(), host_working_exemplars.begin());

		for (int i = 0; i < N; i++) {
			int val = host_working_exemplars[i];
			exemplar_details[val]++;
		}

		int sum_decision_differences = 0;
		for (int i = 0; i < N; i++) {
			int decision_difference = abs(exemplar_details[i] - previous_exemplar_details[i]);
			sum_decision_differences += decision_difference;
		}
		std::cout << "decision differences: " << sum_decision_differences << std::endl;

		previous_exemplar_details = exemplar_details;
		thrust::fill(exemplar_details.begin(), exemplar_details.end(), 0);

		if (sum_decision_differences <= convergence_threshold) {
			algorithm_converged = true;
			break;
		}
		thrust::fill(host_working_exemplars.begin(), host_working_exemplars.end(), 0);
		working_exemplars = host_working_exemplars;
		//make sure to damp 'messages' every cycle
		//damp_messages_launch(responsibility_matrix, availibility_matrix, previous_responsibility_matrix, previous_availibility_matrix, damping_factor, N);
		//previous_responsibility_matrix = responsibility_matrix;
		//previous_availibility_matrix = availibility_matrix;
	}
	std::cout << "exemplars determined, functiuon returning..." << std::endl;
	thrust::copy(working_exemplars.begin(), working_exemplars.end(), exemplars.begin());
}

void CudaInterface::form_similarity_matrix(std::vector<cv::Mat>& input_histograms, cv::Mat& output_similarity_matrix, int N) {

	std::cout << " concatenating histograms..." << std::endl;
	cv::Mat concatenated_histograms(cv::Size(N * 3, 256), input_histograms[0].type());
	cv::Mat* mat_array = input_histograms.data();
	cv::hconcat(mat_array, input_histograms.size(), concatenated_histograms);

	cv::cuda::GpuMat source(concatenated_histograms.size(), concatenated_histograms.type());
	cv::cuda::GpuMat output(output_similarity_matrix.size(), output_similarity_matrix.type());

	source.upload(concatenated_histograms);
	output.upload(output_similarity_matrix);

	std::cout << "forming similarity matrix..." << std::endl;
	form_similarity_matrix_launch(source, output, N);

	output.download(output_similarity_matrix);

	//cudaDeviceReset();
}

void CudaInterface::form_similarity_matrix_color(std::vector<float> &input_colors, cv::Mat& similarity_matrix, int N) {

	cv::cuda::GpuMat output(similarity_matrix.size(), similarity_matrix.type());
	thrust::device_vector<float> source = input_colors;

	std::cout << "forming similarity matrix..." << std::endl;
	form_similarity_matrix_color_launch(source, output, N);
	output.download(similarity_matrix);

	//cudaDeviceReset();
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