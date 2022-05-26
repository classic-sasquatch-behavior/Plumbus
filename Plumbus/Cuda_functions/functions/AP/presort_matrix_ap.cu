#include"../../../cuda_includes.h"
#include"../../../config.h"







__global__ void find_initial_sums(cv::cuda::PtrStepSzf similarity_matrix, cv::cuda::PtrStepSzf availibility_matrix, cv::cuda::PtrStepSzf sum_matrix, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	sum_matrix(row, col) = similarity_matrix(row, col) + availibility_matrix(row, col);
}



__global__ void presort_matrix_ap_kernel(cv::cuda::PtrStepSzf input_matrix, cv::cuda::PtrStepSzf output_matrix, int row_N, int col_N, int carry) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int output_col = (blockIdx.x * blockDim.x) + threadIdx.x;

	//input matrix will always be twice the size of output matrix... but what about carry?
	int input_col_start = output_col * 2;

	if (row >= row_N || output_col >= col_N + carry) { return; }

	if (output_col == col_N + carry - 1) {
		if (carry == 1) {
			output_matrix(row, output_col) = input_matrix(row, input_col_start);
			return;
		}
	}






























}









void presort_matrix_ap_launch(cv::cuda::GpuMat& similarity_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& result, int N) {
	int mat_type = similarity_matrix.type();

	cv::cuda::GpuMat sum_matrix(similarity_matrix.size(), mat_type);
	unsigned int grid_size = (N - (N % 32)) / 32;
	dim3 num_blocks = {grid_size + 1, grid_size + 1, 1};
	dim3 threads_per_block = {32, 32, 1};

	find_initial_sums << <num_blocks, threads_per_block >> > (similarity_matrix, availibility_matrix, sum_matrix, N);
	cudaDeviceSynchronize();

	unsigned int row_N = N;
	unsigned int col_N = N;
	unsigned int new_col_N = 0;
	unsigned int new_col_with_carry = 0;
	bool refine = true;
	unsigned int carry = 0;

	cv::cuda::GpuMat old_output = sum_matrix;

	while (refine) {
		col_N += carry;
		carry = col_N % 2;
		col_N -= carry;
		new_col_N = col_N / 2;
		new_col_with_carry = new_col_N + carry;
	
		cv::cuda::GpuMat new_output(cv::Size(new_col_with_carry, row_N), mat_type);

		unsigned int num_blocks_cols = (new_col_with_carry - (new_col_with_carry % 32))/32;
		num_blocks = {num_blocks_cols + 1, grid_size + 1,1};

		presort_matrix_ap_kernel <<<num_blocks, threads_per_block>>> (old_output, new_output, row_N, col_N, carry);
		cudaDeviceSynchronize();


		if (col_N == 2) { //break out of refine loop and assign result

		}
		else if (col_N == 3) { //trigger exception, then break and assign

		}
		else { //continue with loop
			old_output = new_output;
			new_output.release();
			col_N /= 2;
		}
	}














}