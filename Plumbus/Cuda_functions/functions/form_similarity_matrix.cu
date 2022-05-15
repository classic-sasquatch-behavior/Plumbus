#include"../../cuda_includes.h"







//__device__ void subtract_histograms(cv::cuda::PtrStepSzf &source, cv::cuda::PtrStepSzf &scratch) {
//
//}
//
//__device__ void multiply_histograms(cv::cuda::PtrStepSzf& source, cv::cuda::PtrStepSzf& scratch) {
//
//}









//first attempt: the kernel sums all 256*3 elements
//actually works pretty good. I think this is what they mean by "big kernels" - many threads and many instructions is good
__global__ void form_similarity_matrix_kernel(cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzi dst, int N) {
	int output_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int output_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int source_A_index = output_x;
	int source_B_index = output_y;

	if (output_x >= N || output_y >= N || source_A_index >= N || source_B_index >= N) { return; }

	int A_start_x = 3* source_A_index;
	int B_start_x = 3* source_B_index;

	float sum = 0;

	for (int x_itr = 0; x_itr < 3; x_itr++) {
		int A_x = x_itr + A_start_x;
		int B_x = x_itr + B_start_x;

		for (int y = 0; y < 256; y++) {
			float difference = src(y, A_x) - src(y, B_x);
			float square = difference*difference;
			sum += square;
		}
	}

	int similarity = -(int)round(sum);
	dst(output_y, output_x) = similarity;
}















void form_similarity_matrix_launch(cv::cuda::GpuMat& source, cv::cuda::GpuMat& output, int N) {

	//number of places we have to calculate for: ((N^2)/2) - N
	//each place is a thread
	//we're gonna go sloppy simple and do a thread for each space even though it's unnecessary


	//((N - N%32) / 32) + 1
	int num_blocks_xy = ((N - (N % 32)) / 32) + 1;

	dim3 num_blocks = {(unsigned int)num_blocks_xy, (unsigned int)num_blocks_xy, 1};
	dim3 threads_per_block = {32, 32, 1};

	//cv::cuda::GpuMat scratch(cv::Size(source.rows / 2, source.cols), source.type());


	form_similarity_matrix_kernel <<<num_blocks, threads_per_block>>> (source, output, N);
	cudaDeviceSynchronize();



}