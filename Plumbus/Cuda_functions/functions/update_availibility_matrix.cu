#include"../../cuda_includes.h"









__global__ void update_availibility_matrix_kernel(cv::cuda::PtrStepSzf responsibility_matrix, cv::cuda::PtrStepSzf availibility_matrix, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row >= N || col >= N) { return; }


	float result = 0;

	if (row == col) { //diagonal case

		float column_sum_of_positives = 0;
		for (int row_traversal = 0; row_traversal < N; row_traversal++) {
			if (row_traversal != row) {
				float val_at_cell = responsibility_matrix(row_traversal, col);
				if (val_at_cell > 0) {
					column_sum_of_positives += val_at_cell;
				}
			}
		}
		result = column_sum_of_positives;
	}
	else if (row != col){ //off-diagonal case
		float column_self_responsibility = responsibility_matrix(col, col);
		float column_sum_of_positives = column_self_responsibility;

		for (int col_traversal = 0; col_traversal < N; col_traversal++) {
			if (col_traversal != row && col_traversal != col) {
				float val_at_cell = responsibility_matrix(col_traversal, col);
				if (val_at_cell > 0) {
					column_sum_of_positives += val_at_cell;
				}
			}
		}
		if (column_sum_of_positives < 0) {
			result = column_sum_of_positives;
		}
	}

	availibility_matrix(row, col) = result;
}





void update_availibility_matrix_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibity_matrix, int N) {



	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = { grid_dim_xy,grid_dim_xy,1 };
	dim3 threads_per_block = { block_dim_xy,block_dim_xy,1 };


	update_availibility_matrix_kernel << <num_blocks, threads_per_block >> > (responsibility_matrix, availibity_matrix, N);
	cudaDeviceSynchronize();
}