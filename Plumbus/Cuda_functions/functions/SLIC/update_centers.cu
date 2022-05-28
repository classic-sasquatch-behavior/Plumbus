#include"../../../cuda_includes.h"



__global__ void condense_labels_kernel(iptr labels, iptr row_sums, iptr col_sums, iptr num_sums) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= labels.rows || col >= labels.cols) { return; }

	int label = labels(row, col);

	if (label >= labels.cols) {
		printf("label out of bounds: too high \n");
	}

	if (label < 0) {
		printf("label out of bounds: too low \n");
	}





	atomicAdd(&row_sums(0, label), row);
	atomicAdd(&col_sums(0, label), col);
	atomicAdd(&num_sums(0, label), 1);
}
















//#define cv_ptr_i cv::cuda::PtrStepSzi


__global__ void update_centers_kernel(iptr labels, iptr row_vals, iptr col_vals, iptr row_sums, iptr col_sums, iptr num_sums, int num_centers, int & row_displacement, int & col_displacement) {
	int center_label = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (center_label >= num_centers) { return; }

	int num_rows = row_vals.rows;
	int num_cols = row_vals.cols;
	int col = center_label % num_cols;
	int row = (center_label - col) / num_cols;

	int row_sum = row_sums(0, center_label);
	int col_sum = col_sums(0, center_label);
	int num_constituents = row_sums(0, center_label);
	int row_result = row_sum / num_constituents;
	int col_result = col_sum / num_constituents; 

	int old_row = row_vals(row, col);
	int old_col = col_vals(row, col);
	int row_disp = abs(row_result - old_row);
	int col_disp = abs(col_result - old_col);

	row_vals(row, col) = row_result;
	col_vals(row, col) = col_result;

	atomicAdd(&row_displacement, row_disp);
	atomicAdd(&col_displacement, col_disp);

	//row_displacement(0, center_label) = row_disp;
	//col_displacement(0, center_label) = col_disp;

}

















void update_centers_launch(cv::cuda::GpuMat& labels, cv::cuda::GpuMat& row_vals, cv::cuda::GpuMat& col_vals, int* average_displacement) {

	int center_rows = row_vals.rows;
	int center_cols = row_vals.cols;
	int num_centers = center_rows * center_cols;



	int pixel_rows = labels.rows;
	int pixel_cols = labels.cols;
	int num_pixels = pixel_rows * pixel_cols;

	cv::cuda::GpuMat row_sums(cv::Size(num_centers, 1), CV_32SC1);
	cv::cuda::GpuMat col_sums(cv::Size(num_centers, 1), CV_32SC1);
	cv::cuda::GpuMat num_sums(cv::Size(num_centers, 1), CV_32SC1);



	int block_dim_xy = 32;
	int grid_dim_xy = ((num_pixels - (num_pixels % block_dim_xy)) / block_dim_xy) + 1;

	unsigned int cond_block_dim_x = 32;
	unsigned int cond_block_dim_y = 32;

	unsigned int cond_grid_dim_x = ((pixel_cols - (pixel_cols % cond_block_dim_x)) / cond_block_dim_x) + 1;
	unsigned int cond_grid_dim_y = ((pixel_rows - (pixel_rows % cond_block_dim_y)) / cond_block_dim_y) + 1;







	dim3 cond_num_blocks(cond_grid_dim_x, cond_grid_dim_y, 1);
	dim3 cond_threads_per_block(cond_block_dim_x, cond_block_dim_y, 1);

	std::cout << "condensing labels..." << std::endl;
	condense_labels_kernel << <cond_num_blocks, cond_threads_per_block >> > (labels, row_sums, col_sums, num_sums);
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s: %s \n", cudaGetErrorString(error), "condense labels");
	}



	cv::cuda::GpuMat row_displacement(cv::Size(num_centers, 1), CV_32SC1);
	cv::cuda::GpuMat col_displacement(cv::Size(num_centers, 1), CV_32SC1);
	
	unsigned int block_dim_x = 1024;
	unsigned int grid_dim_x = ((num_centers - (num_centers % block_dim_x)) / block_dim_x) + 1;

	dim3 num_blocks(grid_dim_x, 1, 1);
	dim3 threads_per_block(block_dim_x, 1, 1);

	int sum_row_displacement = 0;
	int sum_col_displacement = 0;

	std::cout << "updating centers (kernel)..." << std::endl;
	update_centers_kernel << <num_blocks, threads_per_block >> > (labels, row_vals, col_vals, row_sums, col_sums, num_sums, num_centers, sum_row_displacement, sum_col_displacement);
	cudaDeviceSynchronize();

	// check for error
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s: %s \n", cudaGetErrorString(error), "update centers");
	}

	sum_row_displacement /= center_rows;
	sum_col_displacement /= center_cols;
	
	int total_displacement = sqrt((center_rows * center_rows) + (center_cols * center_cols));
	*average_displacement = total_displacement;
}