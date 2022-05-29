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

#pragma region superpixels


cv::Mat CudaInterface::SLIC_superpixels(cv::Mat& input, int density, int* num_superpixels_result) {

	const int threshold = 10;

	cv::Mat LAB_src;
	cv::Mat host_labels(input.size(), CV_32SC1 );
	cv::cvtColor(input, LAB_src, cv::COLOR_BGR2Lab);

	int width = input.cols;
	int height = input.rows;
	int num_pixels = width * height;
	int num_superpixels = num_pixels/(density*density); int& N = num_superpixels;
	int superpixel_size = num_pixels / N;
	int grid_interval = sqrt(superpixel_size);

	int SP_rows = height / grid_interval;
	int SP_cols = width / grid_interval;

	


	cv::Mat row_vals( cv::Size(SP_cols, SP_rows), CV_32SC1 );
	cv::Mat col_vals( cv::Size(SP_cols, SP_rows), CV_32SC1 );


	std::cout << "SLIC initializing centers..." << std::endl;
	//initialize centers
	for (int row = 0; row < SP_rows; row++) {
		for (int col = 0; col < SP_cols; col++) {
			int center_row = (row * grid_interval) + round(grid_interval/2);
			int center_col = (col * grid_interval) + round(grid_interval / 2);
			row_vals.at<int>(row, col) = center_row;
			col_vals.at<int>(row, col) = center_col;
		}
	}


	std::cout << "SLIC performing gradient descent..." << std::endl;

	//gradient descent
	for (int row = 0; row < SP_rows; row++) {
		for (int col = 0; col < SP_cols; col++) {
			int focus_row = row_vals.at<int>(row, col);
			int focus_col = col_vals.at<int>(row, col);

			cv::Vec3b focus_color = LAB_src.at<cv::Vec3b>(focus_row, focus_col);
			int gradient[3][3];


			for (int irow = -1; irow <= 1; irow++ ) {
				for (int icol = -1; icol <= 1; icol++) {
					int target_row = focus_row + irow;
					int target_col = focus_col + icol;
					if (target_row < 0 || target_col < 0 || target_row >= height || target_col >= width) { break; }


					int search_vals[4][2] = { {target_row, target_col + 1},{target_row, target_col - 1},{target_row + 1, target_col},{target_row - 1, target_col} };

					cv::Vec3b target_color = LAB_src.at<cv::Vec3b>(target_row, target_col);


					int result_sum = 0;

					for (int i = 0; i < 2; i++) {
						int search_row_pos = search_vals[i * 2][0];
						int search_col_pos = search_vals[i * 2][1];

						int search_row_neg = search_vals[(i* 2) + 1][0];
						int search_col_neg = search_vals[(i * 2) + 1][1];


						if (search_row_pos < 0 || search_col_pos < 0 || search_row_pos >= height || search_col_pos >= width || 
							search_row_neg < 0 || search_col_neg < 0 || search_row_neg >= height || search_col_neg >= width) { break; }
						
						cv::Vec3b search_color_pos = LAB_src.at<cv::Vec3b>(search_row_pos, search_col_pos);

						cv::Vec3b search_color_neg = LAB_src.at<cv::Vec3b>(search_row_neg, search_col_neg);


						for (int channel = 0; channel < 3; channel++) {
							int search_color_pos_channel = search_color_pos[channel];
							int search_color_neg_channel = search_color_neg[channel];

							int channel_result = search_color_pos_channel - search_color_neg_channel;

							result_sum += (channel_result*channel_result);

						}
					}

					gradient[irow + 1][icol + 1] = result_sum;
				}
			}


			int lowest_gradient = INF;
			int lowest_grad_position[2] = { 0,0 };

			for (int irow = 0; irow < 3; irow++) {
				for (int icol = 0; icol < 3; icol++) {
					int this_gradient = gradient[irow][icol];

					if (this_gradient < lowest_gradient) {
						lowest_gradient = this_gradient;
						lowest_grad_position[0] = irow - 1;
						lowest_grad_position[1] = icol - 1;
					}
				}
			}

			int past_row = row_vals.at<int>(row, col);
			int past_col = col_vals.at<int>(row, col);

			int new_row = past_row + lowest_grad_position[0];
			int new_col = past_col + lowest_grad_position[1];

			row_vals.at<int>(row, col) = new_row;
			col_vals.at<int>(row, col) = new_col;
		}
	}







	std::cout << "SLIC creating sector LUT..." << std::endl;


















	//create sector LUT
	cv::Mat sector_LUT(cv::Size(num_superpixels * 9 * 2, 1), CV_32SC1);
	std::vector<std::vector<std::vector<int>>> process_neighbor_coords;

	for (int row = 0; row < SP_rows; row++) {
		for (int col = 0; col < SP_cols; col++) {




			//double check these
			int sector_id = (SP_cols * row) + col;
			//int LUT_index = sector_id * 9 * 2;







			std::vector<std::vector<int>> neighbor_coords;

			for (int irow = -1; irow <= 1; irow++) {
				for (int icol = -1; icol <= 1; icol++) {
					int target_row = row + irow;
					int target_col = col + icol;
					int neighbor_result[2] = {-1, -1};

					if (target_row < 0 || target_col < 0 || target_row >= SP_rows || target_col >= SP_cols) {
						continue;
					}
					else {
						neighbor_result[0] = target_row;
						neighbor_result[1] = target_col;

					}
					neighbor_coords.push_back({ neighbor_result[0], neighbor_result[1] });
				}
			}
			process_neighbor_coords.push_back(neighbor_coords);
		}
	}

	//convert vector to mat


	for (int center = 0; center < process_neighbor_coords.size(); center++) {
		std::vector<std::vector<int>> center_neighbors = process_neighbor_coords[center];
		for (int neighbor = 0; neighbor < center_neighbors.size(); neighbor++) {
			int neighbor_coords[2] = { center_neighbors[neighbor][0], center_neighbors[neighbor][1] };

			sector_LUT.at<int>(0, (center * 9 * 2) + (neighbor*2)) = neighbor_coords[0];
			sector_LUT.at<int>(0, (center * 9 * 2) + (neighbor*2) + 1) = neighbor_coords[1];
		}
	}
































	std::cout << "SLIC uploading mats..." << std::endl;
	cv::cuda::GpuMat d_labels, d_row_vals, d_col_vals, d_sector_LUT;
	//d_src.upload(LAB_src);
	d_labels.upload(host_labels);
	d_row_vals.upload(row_vals);
	d_col_vals.upload(col_vals);
	d_sector_LUT.upload(sector_LUT);

	std::vector<cv::Mat> split_me;
	cv::split(LAB_src, split_me);

	cv::cuda::GpuMat d_src_L, d_src_A, d_src_B;
	d_src_L.upload(split_me[0]);
	d_src_A.upload(split_me[1]);
	d_src_B.upload(split_me[2]);



	std::cout << "SLIC begin main loop..." << std::endl;
	//SLIC algorithm main loop
	bool converged = false;
	while (!converged) {

		std::cout << "SLIC finding labels..." << std::endl;
		find_labels_launch(d_src_L, d_src_A, d_src_B, d_labels, d_row_vals, d_col_vals, d_sector_LUT, density, grid_interval); 

		std::cout << "SLIC updating centers..." << std::endl;
		int average_displacement = 0;
		update_centers_launch(d_labels, d_row_vals, d_col_vals, &average_displacement);

		std::cout << "SLIC average displacement: " << average_displacement << std::endl;
		if (average_displacement <= threshold) {
			converged = true;
		}
	}


	d_labels.download(host_labels);
	*num_superpixels_result = num_superpixels;
	return host_labels;
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