#include"../../cuda_includes.h"
#include"../../config.h"













__global__ void find_row_max_kernel(cv::cuda::PtrStepSzf similarity_matrix, cv::cuda::PtrStepSzf availibility_matrix, cv::cuda::PtrStepSzf AS_max_out, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (row >= N) { return; }


	float AS_max = -INF;
	float AS_second_max = -INF;



	for (int col = 0; col < N; col++) {
		float A_val_at_col = availibility_matrix(row, col);
		float S_val_at_col = similarity_matrix(row, col);
		float AS_val_at_col = A_val_at_col + S_val_at_col;


		//check AS
		if (AS_val_at_col > AS_second_max) {
			AS_second_max = AS_val_at_col;
		}
		if (AS_val_at_col > AS_max) {
			AS_second_max = AS_max;
			AS_max = AS_val_at_col;
		}
	}

	AS_max_out(row, 0) = AS_max;
	AS_max_out(row, 1) = AS_second_max;


}





__global__ void calculate_off_diagonal_kernel(cv::cuda::PtrStepSzf similarity_matrix, cv::cuda::PtrStepSzf availibility_matrix, cv::cuda::PtrStepSzf AS_max, cv::cuda::PtrStepSzf responsibility_matrix,int N) {
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }



	float this_similarity_val = similarity_matrix(row, col);
	float this_availibility_val = availibility_matrix(row, col);
	float this_AS_val = this_similarity_val + this_availibility_val;

	float AS_row_max_first = AS_max(row, 0);
	float AS_row_max_second = AS_max(row, 1);

	float subtrahend = 0;

	if (this_AS_val != AS_row_max_first) {
		subtrahend = AS_row_max_first;
	}
	else {
		subtrahend = AS_row_max_second;
	}

	float result = this_similarity_val - subtrahend;
	responsibility_matrix(row, col) = result;
}











void update_responsibility_matrix_launch(cv::cuda::GpuMat& similarity_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& responsibility_matrix, int N) {



	//find row max prep
	unsigned int find_row_max_block_dim_y = 1024;
	unsigned int find_row_max_grid_dim_y = ((N - (N % 1024))/1024) + 1;

	dim3 find_row_max_num_blocks{ 1, find_row_max_grid_dim_y, 1 };
	dim3 find_row_max_threads_per_block {1, find_row_max_block_dim_y, 1};

	cv::cuda::GpuMat AS_max(cv::Size(2, N), similarity_matrix.type());

	//claculate off-diagonal prep

	unsigned int off_diagonal_block_dim_xy = 32;
	unsigned int off_diagonal_grid_dim_xy = ((N - (N % 32)) / 32) + 1;

	dim3 off_diagonal_num_blocks = { off_diagonal_grid_dim_xy,off_diagonal_grid_dim_xy,1 };
	dim3 off_diagonal_threads_per_block = { off_diagonal_block_dim_xy,off_diagonal_block_dim_xy,1 };






	find_row_max_kernel << <find_row_max_num_blocks, find_row_max_threads_per_block >> > (similarity_matrix, availibility_matrix, AS_max, N);
	cudaDeviceSynchronize();

	calculate_off_diagonal_kernel << <off_diagonal_num_blocks, off_diagonal_threads_per_block >> > (similarity_matrix, availibility_matrix, AS_max, responsibility_matrix, N);
	cudaDeviceSynchronize();

}