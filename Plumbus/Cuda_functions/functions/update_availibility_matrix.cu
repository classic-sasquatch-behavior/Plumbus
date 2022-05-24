#include"../../cuda_includes.h"







__global__ void availibility_matrix_sum_columns_kernel(cv::cuda::PtrStepSzf responsibility_matrix, cv::cuda::PtrStepSzf column_sums, int N) {

	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (col >= N) { return; }


	int column_sum = 0;
	for (int row = 0; row < N; row++) {
		int row_val = responsibility_matrix(row, col);
		if (row_val > 0) {
			column_sum += row_val;
		}
	}
	//printf("%d \n", column_sum);
	column_sums(0, col) = column_sum;
}


__global__ void availibility_matrix_calculate_off_diagonal(cv::cuda::PtrStepSzf responsibility_matrix, cv::cuda::PtrStepSzf column_sums, cv::cuda::PtrStepSzf availibility_matrix, int N) {
	
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row >= N || col >= N || row == col) { return; }
	

	int responsibility_at_pair = responsibility_matrix(row, col);
	int self_responsibility = responsibility_matrix(col, col);
	int responsibility_col_sum = column_sums(0,col);
	int result = 0;

	if (responsibility_at_pair > 0) {
		responsibility_col_sum -= responsibility_at_pair;
	}
	if (self_responsibility > 0) {
		responsibility_col_sum -= self_responsibility;
	}

	result = responsibility_col_sum + self_responsibility;
	if (result > 0) {
		result = 0;
	}

	availibility_matrix(row, col) = result;
}





__global__ void availibility_matrix_calculate_diagonal(cv::cuda::PtrStepSzf responsibility_matrix, cv::cuda::PtrStepSzf column_sums, cv::cuda::PtrStepSzf availibility_matrix, int N) {
	int diag = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (diag >= N) { return; }

	int self_responsibility = responsibility_matrix(diag, diag);
	int responsibility_col_sum = column_sums(0, diag);
	//printf("%d \n", responsibility_col_sum);

	if (self_responsibility > 0) {
		responsibility_col_sum -= self_responsibility;
	}

	availibility_matrix(diag, diag) = responsibility_col_sum;
}












void update_availibility_matrix_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibity_matrix, int N) {


	//sum columns prep
	
	unsigned int sum_columns_block_dim_x = 1024;
	unsigned int sum_columns_grid_dim_x = ((N - (N % 1024)) / 1024) + 1;

	dim3 sum_columns_num_blocks = {sum_columns_grid_dim_x,1,1};
	dim3 sum_columns_threads_per_block = {sum_columns_block_dim_x,1,1};

	cv::cuda::GpuMat column_sums(cv::Size(N, 1), responsibility_matrix.type());


	//off-diagonal prep

	unsigned int calculate_off_diagonal_grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int calculate_off_diagonal_block_dim_xy = 32;

	dim3 calculate_off_diagonal_num_blocks = { calculate_off_diagonal_grid_dim_xy,calculate_off_diagonal_grid_dim_xy,1 };
	dim3 calculate_off_diagonal_threads_per_block = { calculate_off_diagonal_block_dim_xy,calculate_off_diagonal_block_dim_xy,1 };


	//on-diagonal prep

	unsigned int calculate_on_diagonal_grid_dim_x = 1024;
	unsigned int calculate_on_diagonal_block_dim_x = ((N - (N % 1024)) / 1024) + 1;
	dim3 calculate_on_diagonal_num_blocks = { calculate_on_diagonal_grid_dim_x ,1,1 };
	dim3 calculate_on_diagonal_threads_per_block = { calculate_on_diagonal_block_dim_x,1,1 };


	availibility_matrix_sum_columns_kernel << <sum_columns_num_blocks, sum_columns_threads_per_block >> > (responsibility_matrix, column_sums, N);
	cudaDeviceSynchronize();

	availibility_matrix_calculate_off_diagonal << <calculate_off_diagonal_num_blocks, calculate_off_diagonal_threads_per_block >> > (responsibility_matrix, column_sums, availibity_matrix, N);
	availibility_matrix_calculate_diagonal << <calculate_on_diagonal_num_blocks, calculate_on_diagonal_threads_per_block >> > (responsibility_matrix, column_sums, availibity_matrix, N);
	cudaDeviceSynchronize();














}