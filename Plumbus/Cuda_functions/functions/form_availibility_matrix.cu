#include"../../cuda_includes.h"









__global__ void form_availibility_matrix_kernel(cv::cuda::PtrStepSzi src, cv::cuda::PtrStepSzi dst, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (row >= N || col >= N) { return; }


	int result = 0;

	if (row == col) { //diagonal

		int column_sum_of_positives = 0;
		for (int col_traversal = 0; col_traversal < N; col_traversal++) {
			if (col_traversal != row) {
				int val_at_cell = src(col_traversal, col);
				if (val_at_cell > 0) {
					column_sum_of_positives += val_at_cell;
				}
			}
		}
		result = column_sum_of_positives;
	}
	else { //off-diagonal
		int column_self_responsibility = src(col, col);
		int column_sum_of_positives = column_self_responsibility;

		for (int col_traversal = 0; col_traversal < N; col_traversal++) {
			if (col_traversal != row && col_traversal != col) {
				int val_at_cell = src(col_traversal, col);
				if (val_at_cell > 0) {
					column_sum_of_positives += val_at_cell;
				}
			}
		}
		result = column_sum_of_positives;
	}

	dst(row, col) = result;
	//diagonal: sum of the positive responsibilities of the COLUMMN, excluding the self responsibility
	//off-diagonal: COLUMN self-responsibility + sum of positive responsibilities in the column excluding responsibility of this exact row/col
}





void form_availibility_matrix_launch(cv::cuda::GpuMat& source, cv::cuda::GpuMat& output, int N) {



	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = { grid_dim_xy,grid_dim_xy,1 };
	dim3 threads_per_block = { block_dim_xy,block_dim_xy,1 };


	form_availibility_matrix_kernel << <num_blocks, threads_per_block >> > (source, output, N);
	cudaDeviceSynchronize();
}