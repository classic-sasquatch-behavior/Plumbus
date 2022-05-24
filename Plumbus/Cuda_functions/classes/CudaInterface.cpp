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


void CudaInterface::affinity_propagation_color(cv::Mat& colors, cv::Mat &coordinates, cv::Mat& exemplars, int N) {



	std::cout << "running affinity propagation..." << std::endl;

	double lowest_val;
	double highest_val;
	cv::Size matrix_size(N, N);
	const int matrix_type = CV_32FC1;

	const float preference_factor = 2;
	const float damping_factor = 0.8f;
	const int max_cycles = 10000;
	const int max_const_cycles = 10;
	const int const_threshold = 10;



	cv::cuda::GpuMat similarity_matrix(matrix_size, matrix_type, cv::Scalar{0});
	cv::cuda::GpuMat responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat critereon_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });

	cv::cuda::GpuMat previous_responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat previous_availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });



	cv::cuda::GpuMat color_source(cv::Size(3, N), CV_32FC1); 
	cv::cuda::GpuMat coordinate_source(cv::Size(2, N), CV_32FC1);
	cv::cuda::GpuMat working_exemplars(cv::Size(1, N), CV_32SC1, cv::Scalar{0}); 
	
	color_source.upload(colors);
	coordinate_source.upload(coordinates);









	cv::Mat exemplar_differences_through_time(cv::Size(1, N), CV_32SC1, cv::Scalar(0));
	cv::Mat host_exemplars(cv::Size(1, N), CV_32SC1);
	cv::Mat exemplar_differences(cv::Size(1, N), CV_32SC1, cv::Scalar(0));
	cv::Mat previous_exemplar_differences(cv::Size(1, N), CV_32SC1, cv::Scalar(0));







	
	std::cout << "forming similarity matrix..." << std::endl;

	form_similarity_matrix_color_launch(color_source, coordinate_source, similarity_matrix, N);



	//can get speedup by making this cuda
	cv::Mat h_sim_mat(matrix_size, matrix_type);
	similarity_matrix.download(h_sim_mat);
	cv::Mat similarity_matrix_diagonal = h_sim_mat.diag(0);
	cv::minMaxIdx(h_sim_mat, &lowest_val, &highest_val);
	float similarity_min = lowest_val * preference_factor; //pretty good: lowest_val * 2 (from lowest_val * 1)
	similarity_matrix_diagonal.setTo(similarity_min);
	similarity_matrix.upload(h_sim_mat);
	util->print_gpu_mat(similarity_matrix, 5);

	bool algorithm_converged = false;
	int cycles = 0;
	int const_cycles = 0;
	bool initialize = true;


	while (!algorithm_converged) {
		std::cout << "begin cycle " << cycles << std::endl;

		std::cout << "updating responsibility matrix..." << std::endl;
		update_responsibility_matrix_launch(similarity_matrix, availibility_matrix, responsibility_matrix, N);
		if (initialize) {
			initialize = false;
			previous_responsibility_matrix = responsibility_matrix;
		}
		dampen_messages_launch(previous_responsibility_matrix, responsibility_matrix, damping_factor, N); 
		previous_responsibility_matrix = responsibility_matrix;
		util->print_gpu_mat(responsibility_matrix, 5);




		//begin something fucked up
		std::cout << "updating availibility matrix..." << std::endl;
		update_availibility_matrix_launch(responsibility_matrix, availibility_matrix, N); 
		util->print_gpu_mat(availibility_matrix, 5);
		dampen_messages_launch(previous_availibility_matrix, availibility_matrix, damping_factor, N);
		previous_availibility_matrix = availibility_matrix;
		util->print_gpu_mat(availibility_matrix, 5);
		//end something fucked up






		std::cout << "updating critereon matrix..." << std::endl;
		update_critereon_matrix_launch(responsibility_matrix, availibility_matrix, critereon_matrix, N);
		util->print_gpu_mat(critereon_matrix, 5);

		std::cout << "extracting exemplars..." << std::endl;
		extract_exemplars_launch(critereon_matrix, working_exemplars, N);
		working_exemplars.download(host_exemplars);

		for (int i = 0; i < N; i++) {
			exemplar_differences.at<int>(i, 0) = 0;
		}
		for (int i = 0; i < N; i++) {
			int exemplar_of_i = host_exemplars.at<int>(i, 0);
			exemplar_differences.at<int>(exemplar_of_i, 0) += 1;
			//std::cout << "exemplar of i: " << exemplar_of_i << std::endl;
		}
		for (int i = 0; i < N; i++) {
			int val_a = exemplar_differences.at<int>(i, 0);
			int val_b = previous_exemplar_differences.at<int>(i, 0);
			exemplar_differences_through_time.at<int>(i, 0) = val_a - val_b;

		}
		int difference_sum = 0;
		for (int row = 0; row < N; row++) {
			int val_at = exemplar_differences_through_time.at<int>(row, 0);
			difference_sum += abs(val_at);
			
		}
		for (int i = 0; i < N; i++) {
			previous_exemplar_differences.at<int>(i, 0) = exemplar_differences.at<int>(i, 0);
		}
		std::cout << "change in exemplars found: " << difference_sum << std::endl;
		//std::cout << "end cycle " << cycles << std::endl << std::endl;
		//std::cout << "----------------" << std::endl << std::endl;
		cycles++;




		//if (difference_sum <= const_threshold || double_derivative <= const_threshold) {
		if (difference_sum <= const_threshold) {
			const_cycles++;
		}
		else {
			const_cycles = 0;
		}
		std::cout << "const cycles: " << const_cycles << std::endl;


		if (cycles >= max_cycles || const_cycles >= max_const_cycles) {
			algorithm_converged = true;
			std::cout << "affintiy propagation ended at " << cycles << " cycles" << std::endl;
			extract_exemplars_launch(critereon_matrix, working_exemplars, N);
			working_exemplars.download(host_exemplars);
			working_exemplars.download(exemplars);
		}
	}

	std::cout << "exemplars determined, function returning..." << std::endl;







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