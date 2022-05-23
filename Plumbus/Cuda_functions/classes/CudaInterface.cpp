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


void CudaInterface::affinity_propagation_color(cv::Mat& colors, cv::Mat& exemplars, int N) {



	std::cout << "running affinity propagation..." << std::endl;

	cv::Size matrix_size(N, N);
	float damping_factor = 0.9f;
	int convergence_threshold = 10;
	//int num_static_cycles_before_convergence = 3;
	int max_cycles = 10;
	int matrix_type = CV_32FC1;


	cv::cuda::GpuMat similarity_matrix(matrix_size, matrix_type, cv::Scalar{0});
	cv::cuda::GpuMat responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat critereon_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });

	cv::cuda::GpuMat previous_responsibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });
	cv::cuda::GpuMat previous_availibility_matrix(matrix_size, matrix_type, cv::Scalar{ 0 });


	//replace thrust with opencv
	cv::cuda::GpuMat source(cv::Size(3, N), CV_32FC1); //actually, you should pass in input colors as a mat.
	cv::cuda::GpuMat working_exemplars(cv::Size(1, N), CV_32SC1, cv::Scalar{0}); //likewise, you should pass working exemplars out as mat
	
	source.upload(colors);








	cv::Mat host_working_exemplars(cv::Size(1, N), CV_32SC1);
	cv::Mat exemplar_details(cv::Size(1, N), CV_32SC1);
	cv::Mat previous_exemplar_details(cv::Size(1, N), CV_32SC1);










	
	std::cout << "forming similarity matrix..." << std::endl;

	form_similarity_matrix_color_launch(source, similarity_matrix, N);



	//can get speedup by making this cuda
	cv::Mat h_sim_mat(matrix_size, matrix_type);
	similarity_matrix.download(h_sim_mat);
	cv::Mat similarity_matrix_diagonal = h_sim_mat.diag(0);
	double lowest_val;
	double highest_val;
	cv::minMaxIdx(h_sim_mat, &lowest_val, &highest_val);

	float similarity_min = lowest_val * 2; //pretty good: lowest_val * 2 (from lowest_val * 1)

	similarity_matrix_diagonal.setTo(similarity_min);
	similarity_matrix.upload(h_sim_mat);
	util->print_gpu_mat(similarity_matrix, 5);

	bool algorithm_converged = false;
	bool initialize = true;
	//int cycles_without_change = 0;
	int cycles = 0;



	while (!algorithm_converged) {
		std::cout << "begin cycle " << cycles << std::endl;






		std::cout << "updating responsibility matrix..." << std::endl;
		update_responsibility_matrix_launch(similarity_matrix, availibility_matrix, responsibility_matrix, N);
		if(initialize){
			previous_responsibility_matrix = responsibility_matrix;
			previous_availibility_matrix = availibility_matrix;
			initialize = false;
		}
		dampen_messages_launch(previous_responsibility_matrix, responsibility_matrix, damping_factor, N); 
		previous_availibility_matrix = responsibility_matrix;
		util->print_gpu_mat(responsibility_matrix, 5);






		std::cout << "updating availibility matrix..." << std::endl;
		update_availibility_matrix_launch(responsibility_matrix, availibility_matrix, N); 
		dampen_messages_launch(previous_availibility_matrix, availibility_matrix, damping_factor, N);
		previous_availibility_matrix = availibility_matrix;
		util->print_gpu_mat(availibility_matrix, 5);






		std::cout << "updating critereon matrix..." << std::endl;
		update_critereon_matrix_launch(responsibility_matrix, availibility_matrix, critereon_matrix, N);
		util->print_gpu_mat(critereon_matrix, 5);








		std::cout << "end cycle " << cycles << std::endl << std::endl;
		std::cout << "----------------" << std::endl << std::endl;
		cycles++;

		if (cycles >= max_cycles) {
			algorithm_converged = true;
			std::cout << "extracting exemplars..." << std::endl;
			extract_exemplars_launch(critereon_matrix, working_exemplars, N);
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