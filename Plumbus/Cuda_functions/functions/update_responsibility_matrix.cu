#include"../../cuda_includes.h"
#include"../../config.h"






__global__ void update_responsibility_matrix_kernel(cv::cuda::PtrStepSzf similarity_matrix, cv::cuda::PtrStepSzf availibility_matrix, cv::cuda::PtrStepSzf responsibility_matrix, int N) {
	
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	float max_similarity = -INF;
	float source_val = similarity_matrix(row, col);
	float result = 0;

	if (row == col) { //diagonal case
		for (int col_iterator = 0; col_iterator < N; col_iterator++) {
			if (col_iterator != col) {
				float similarity = similarity_matrix(row, col_iterator);
				if (similarity > max_similarity) {
					max_similarity = similarity;
				}
			}
		}
		result = source_val - max_similarity;
	}

	else if (row != col) { //off-diagonal case
		for (int col_iterator = 0; col_iterator < N; col_iterator++) {
			if (col_iterator != col) {
				float similarity = similarity_matrix(row, col_iterator);
				float availibility = availibility_matrix(row, col_iterator);

				float s_a_sum = similarity + availibility;
				if (s_a_sum > max_similarity) {
					max_similarity = s_a_sum;
				}
			}
		}
		result = source_val - max_similarity;
	}

	responsibility_matrix(row, col) = result;
}












void update_responsibility_matrix_launch(cv::cuda::GpuMat& similarity_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& responsibility_matrix, int N) {

	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = {grid_dim_xy,grid_dim_xy,1};
	dim3 threads_per_block = {block_dim_xy,block_dim_xy,1};


	update_responsibility_matrix_kernel << <num_blocks, threads_per_block >> > (similarity_matrix, availibility_matrix, responsibility_matrix, N);
	cudaDeviceSynchronize();


}