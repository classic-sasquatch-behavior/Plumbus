#include"../../cuda_includes.h"
#include"../../config.h"





__global__ void dampen_messages_kernel(cv::cuda::PtrStepSzf previous_matrix, cv::cuda::PtrStepSzf updated_matrix, cv::cuda::PtrStepSzf resulting_matrix, float damping_factor, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	float previous_val = previous_matrix(row, col);
	float updated_val = previous_matrix(row, col);

	float result = (damping_factor * previous_val) + ((1 - damping_factor) * (updated_val));
	resulting_matrix(row, col) = result;
}












void dampen_messages_launch(cv::cuda::GpuMat& previous_matrix, cv::cuda::GpuMat& updated_matrix, float damping_factor, int N) {


	unsigned int block_dim_xy = 32;
	unsigned int grid_dim_xy = ((N - (N % block_dim_xy)) / block_dim_xy) + 1;

	dim3 num_blocks = {grid_dim_xy, grid_dim_xy, 1};
	dim3 threads_per_block = {block_dim_xy, block_dim_xy, 1};

	cv::cuda::GpuMat resulting_matrix(updated_matrix.size(), updated_matrix.type());

	dampen_messages_kernel << <num_blocks, threads_per_block >> > (previous_matrix, updated_matrix, resulting_matrix, damping_factor, N);
	cudaDeviceSynchronize();

	updated_matrix = resulting_matrix;
}