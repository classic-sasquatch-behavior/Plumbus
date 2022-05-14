#include"../../cuda_includes.h"







__device__ void subtract_histograms(cv::cuda::PtrStepSzf &source, cv::cuda::PtrStepSzf &scratch) {

}







__device__ void multiply_histograms(cv::cuda::PtrStepSzf& source, cv::cuda::PtrStepSzf& scratch) {

}









//first attempt: the kernel sums all 256*3 elements
__global__ void form_similarity_matrix_kernel(cv::cuda::PtrStepSzf source, cv::cuda::PtrStepSzf scratch, cv::cuda::PtrStepSzi output, int N) {
	int output_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int output_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int source_A_index = output_x;
	int source_B_index = output_y;

	if (output_x > N || output_y > N || source_A_index > N || source_B_index > N) { return; }


















}















void form_similarity_matrix_launch(cv::cuda::GpuMat& source, cv::cuda::GpuMat& output, int N) {

	//number of places we have to calculate for: ((N^2)/2) - N
	//each place is a thread
	//we're gonna go sloppy simple and do a thread for each space even though it's unnecessary


	//((N - N%32) / 32) + 1
	int num_blocks_xy = ((N - (N % 32)) / 32) + 1;

	dim3 num_blocks = {num_blocks_xy, num_blocks_xy, 1};
	dim3 threads_per_block = {32, 32, 1};

	cv::cuda::GpuMat scratch(source.size(), source.type());


	form_similarity_matrix_kernel <<<num_blocks, threads_per_block>>> (source, scratch, output, N);
	cudaDeviceSynchronize();



}