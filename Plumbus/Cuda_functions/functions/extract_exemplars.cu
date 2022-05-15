#include"../../cuda_includes.h"
#include"../../config.h"










__global__ void extract_exemplars_kernel(cv::cuda::PtrStepSzi input, int* output, int N ) {
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;


	if (row >= N) {return;}


	int highest_val = -INF;
	int row_exemplar = 0;

	for (int col_iterator = 0; col_iterator < N; col_iterator++) {
		
		int val_at_col = input(row, col_iterator);
		if (val_at_col > highest_val) {
			highest_val = val_at_col;
			row_exemplar = col_iterator;
		}
	}

	output[row] = row_exemplar;
}








void extract_exemplars_launch(cv::cuda::GpuMat& input, thrust::device_vector<int> &output, int N) {

	unsigned int grid_size = ((N - (N % 32)) / 32) + 1;

	dim3 num_blocks = { 1, grid_size, 1 };
	dim3 threads_per_block = {1, 32, 1};

	int* d_output = thrust::raw_pointer_cast(output.data());


	extract_exemplars_kernel << <num_blocks, threads_per_block >> > (input, d_output, N);

}



















