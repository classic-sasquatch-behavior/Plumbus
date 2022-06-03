#include"../../../cuda_includes.h"
#include"../../../config.h"





__global__ void sum_vec_kernel(iptr input, iptr working_sums, int N, int carry) {

	int out_pos = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (out_pos >= N) { return; }

	if (carry == 1) {
		if (out_pos == N - 1) {
			working_sums(0, out_pos) = input(0, out_pos * 2);
		}
		else {
			int sum_start = out_pos * 2;

			int val_a = input(0, sum_start);
			int val_b = input(0, sum_start + 1);

			working_sums(0, out_pos) = val_a + val_b;
		}
	}
	else {
		int sum_start = out_pos * 2;

		int val_a = input(0, sum_start);
		int val_b = input(0, sum_start + 1);

		working_sums(0, out_pos) = val_a + val_b;
	}
}











void sum_vec_launch(gMat& input, int* sum) {
	//vector is assumed to be 1d and horizontal
	//this is exclusive sum

	int N = input.cols;

	int carry = 0;

	int previous_N = N;
	int temp_N = -1;

	gMat previous_sums = input;



	while (temp_N != 1) {

		carry = previous_N % 2;
		temp_N = ((previous_N - carry) / 2) + carry;

		gMat working_sums(cv::Size(temp_N, 1), CV_32SC1);

		unsigned int block_dim_x = 1024;
		unsigned int grid_dim_x = ((temp_N - (temp_N % block_dim_x)) / block_dim_x) + 1;

		dim3 num_blocks(grid_dim_x, 1, 1);
		dim3 threads_per_block(block_dim_x, 1, 1);


		sum_vec_kernel << <num_blocks, threads_per_block >> > (previous_sums, working_sums, temp_N, carry);

		previous_N = temp_N;
		previous_sums = working_sums;

	}

	cv::Mat total_sum;
	previous_sums.download(total_sum);

	int final_sum = total_sum.at<int>(0, 0);

	sum = &final_sum;
}