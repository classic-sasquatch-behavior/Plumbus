#include"../../cuda_includes.h"
#include"../../config.h"










__global__ void extract_exemplars_kernel(cv::cuda::PtrStepSzf input, cv::cuda::PtrStepSzi output, int N ) { //seemingly broken behavior might arise from confusion between col and row here
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;


	if (row >= N) {return;}


	float highest_val = -INF;
	int row_exemplar = 0;

	for (int col_iterator = 0; col_iterator < N; col_iterator++) {
		
		float val_at_col = input(row, col_iterator);
		if (val_at_col > highest_val) {
			highest_val = val_at_col;
			row_exemplar = col_iterator;
		}
	}

	output(0, row) = row_exemplar;
}








void extract_exemplars_launch(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, int N) {


	unsigned int grid_size = ((N - (N % 1024)) / 1024) + 1;

	dim3 num_blocks = { grid_size, 1024, 1 };
	dim3 threads_per_block = {1024, 1, 1};



	extract_exemplars_kernel << <num_blocks, threads_per_block >> > (input, output, N);

}



















