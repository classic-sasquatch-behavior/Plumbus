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

	//ok this part should be good
	const int threshold = 10;

	cv::Mat LAB_src;
	cv::Mat host_labels(input.size(), CV_32SC1 );
	cv::cvtColor(input, LAB_src, cv::COLOR_BGR2Lab);

	int pixel_rows = input.rows;
	int pixel_cols = input.cols;
	int num_pixels = pixel_rows * pixel_cols; int& N = num_pixels;
	int num_superpixels = N/(density*density); int& K = num_superpixels; //in general, this is a sus way to do this. we'll allow it for now, though
	int superpixel_size = N/K;
	int grid_interval = sqrt(superpixel_size); int& S = grid_interval;

	int SP_rows = pixel_rows / S;
	int SP_cols = pixel_cols / S;

	cv::Mat row_vals( cv::Size(K, 1), CV_32SC1 );
	cv::Mat col_vals( cv::Size(K, 1), CV_32SC1 );

	//should be good
	std::cout << "SLIC initializing centers..." << std::endl;
	//initialize centers
	for (int row = 0; row < SP_rows; row++) {
		for (int col = 0; col < SP_cols; col++) {
			int center_row = (row * grid_interval) + round(grid_interval/2); //if grid interval isn't mathematically perfect (in the sense I'm assuming it is), then it could be leading to oob errors
			int center_col = (col * grid_interval) + round(grid_interval/2);
			int center_id = (row * SP_cols ) + col;

			row_vals.at<int>(0, center_id) = center_row;
			col_vals.at<int>(0, center_id) = center_col;
		}
	}
























	std::cout << "SLIC performing gradient descent..." << std::endl;











	//god this whole section is a mess. think I'll have to break out the clipboard

	//gradient descent
	for (int center = 0; center < K; center++) {
			int focus_row = row_vals.at<int>(0, center);
			int focus_col = col_vals.at<int>(0, center);

			cv::Vec3b focus_color = LAB_src.at<cv::Vec3b>(focus_row, focus_col);
			int gradient[3][3]; //why?





			for (int irow = -1; irow <= 1; irow++ ) {
				for (int icol = -1; icol <= 1; icol++) {
					int target_row = focus_row + irow;
					int target_col = focus_col + icol;
					if (target_row < 0 || target_col < 0 || target_row >= pixel_rows|| target_col >= pixel_cols) { break; }

					//why?
					int search_vals[4][2] = { {target_row, target_col + 1},{target_row, target_col - 1},{target_row + 1, target_col},{target_row - 1, target_col} }; 

					//I'm not seeing where we use this
					cv::Vec3b target_color = LAB_src.at<cv::Vec3b>(target_row, target_col);









					int result_sum = 0;

					for (int i = 0; i < 2; i++) { //why does this go twice?
						int search_row_pos = search_vals[i * 2][0];
						int search_col_pos = search_vals[i * 2][1];

						int search_row_neg = search_vals[(i* 2) + 1][0];
						int search_col_neg = search_vals[(i * 2) + 1][1];


						if (search_row_pos < 0 || search_col_pos < 0 || search_row_pos >= height || search_col_pos >= width || 
							search_row_neg < 0 || search_col_neg < 0 || search_row_neg >= height || search_col_neg >= width) { break; }
						//actually, this might be too much.




						
						cv::Vec3b search_color_pos = LAB_src.at<cv::Vec3b>(search_row_pos, search_col_pos);
						cv::Vec3b search_color_neg = LAB_src.at<cv::Vec3b>(search_row_neg, search_col_neg);
						//what?




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

			int new_row = past_row + lowest_grad_position[0]; //this is wrong
			int new_col = past_col + lowest_grad_position[1];

			row_vals.at<int>(row, col) = new_row;
			col_vals.at<int>(row, col) = new_col;
		
	}



	//end hot mess



















	//should be good now. doenst store sector coords anymore, just the ids. now, put everything else in terms of ids rather than sector coords.
	std::cout << "SLIC creating sector LUT..." << std::endl;
	//create sector LUT
	cv::Mat sector_LUT(cv::Size(num_superpixels * 9, 1), CV_32SC1);
	std::vector<std::vector<int>> process_neighbor_ids;

	//initialize vector sizes
	for (int focus_center = 0; focus_center < K; focus_center++) {
		std::vector<int> new_ids;
		for (int target_center = 0; target_center < 9; target_center++) {
			new_ids.push_back(-1);
		}
		process_neighbor_ids.push_back(new_ids);
	}

	for (int row = 0; row < SP_rows; row++) {
		for (int col = 0; col < SP_cols; col++) {

			int focus_sector_id = (row* SP_cols) + col;

			int center = 0;
			for (int irow = -1; irow <= 1; irow++) {
				for (int icol = -1; icol <= 1; icol++) {
					int target_row = row + irow;
					int target_col = col + icol;
					int target_sector_id = (target_row * SP_cols) + target_col;

					//this is just a stupid way to do it, change it
					if (target_row < 0 || target_col < 0 || target_row >= SP_rows || target_col >= SP_cols) {
						continue;
					}
					else {
						process_neighbor_ids[focus_sector_id][center] = target_sector_id;
					}
					center++;

				}
			}

		}
	}

	//convert to mat
	for (int center = 0; center < K; center++) {
		for (int neighbor = 0; neighbor < 9; neighbor++) {

			sector_LUT.at<int>(0, (center * 9) + neighbor) = process_neighbor_ids[center][neighbor];
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