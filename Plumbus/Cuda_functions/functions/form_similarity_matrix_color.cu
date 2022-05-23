#include"../../cuda_includes.h"















__global__ void form_similarity_matrix_color_kernel(cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int N) {
	int output_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int output_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//int source_A_index = output_x * 3;
	//int source_B_index = output_y * 3;

	if (output_x >= N || output_y >= N) { return; }

	float sum = 0;

	for (int channel = 0; channel < 3; channel++) {

		float val_a = src(output_x, channel);
		float val_b = src(output_y, channel);

		float difference = val_a - val_b;

		float square = difference*difference;
		//float square = abs(difference);
		// 
		//sum += sqrt(square);
		sum += square;

	}

	float similarity = -sqrt(sum);
	dst(output_y, output_x) = similarity;
}




__global__ void form_similarity_matrix_coordinate_kernel(cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int N) {
	int output_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int output_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//int source_A_index = output_x * 3;
	//int source_B_index = output_y * 3;

	if (output_x >= N || output_y >= N) { return; }

	float sum = 0;

	for (int channel = 0; channel < 2; channel++) {

		float val_a = src(output_x, channel);
		float val_b = src(output_y, channel);

		float difference = val_a - val_b;

		float square = difference * difference;
		//float square = abs(difference);
		// 
		//sum += sqrt(square);
		sum += square;

	}

	float similarity = -sqrt(sum);
	dst(output_y, output_x) += similarity;






}










void form_similarity_matrix_color_launch(cv::cuda::GpuMat& color_source, cv::cuda::GpuMat &coordinate_source, cv::cuda::GpuMat& output, int N) {



	int num_blocks_xy = ((N - (N % 32)) / 32) + 1;

	dim3 num_blocks = { (unsigned int)num_blocks_xy, (unsigned int)num_blocks_xy, 1 };
	dim3 threads_per_block = { 32, 32, 1 };


	form_similarity_matrix_color_kernel << <num_blocks, threads_per_block >> > (color_source, output, N);
	cudaDeviceSynchronize();

	form_similarity_matrix_coordinate_kernel << <num_blocks, threads_per_block >> > (coordinate_source, output, N);
	cudaDeviceSynchronize();

}