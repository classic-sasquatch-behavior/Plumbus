#include"../../cuda_includes.h"










__global__ void selective_blur_kernel(uchar* src, int step, uchar* dst, int threshold, int width, int height, int substep, int kernel_measure) {

	__shared__ int num_valid_neighbors;
	__shared__ int BGR_sum[3];

	const int focus_col = blockIdx.x;
	const int focus_row = blockIdx.y;


	const int target_col = focus_col + threadIdx.x - kernel_measure;
	const int target_row = focus_row + threadIdx.y - kernel_measure;
	const int target_channel = threadIdx.z;
	num_valid_neighbors = 0;

	if ((focus_row >= height) || (focus_col >= width) || (target_row >= height) || (target_col >= width) || (target_row < 0) || (target_col < 0)) { return; }


	int focus_address = (focus_row * step) + (focus_col * substep * 3);
	int focus_color = src[focus_address + (target_channel * substep)];
	
	int target_address = (target_row * step) + (target_col * substep * 3);
	int target_color = src[target_address + (target_channel * substep)];

	int difference = focus_color - target_color;
	if (difference < 0) {
		difference = difference * -1;
	}

	if (difference <= threshold) {
		atomicAdd(BGR_sum + target_channel, target_color);
		atomicAdd(&num_valid_neighbors, 1);
	}

	__syncthreads();

	if (focus_row == target_row && focus_col == target_col) {
		dst[focus_address + (target_channel * substep)] = BGR_sum[target_channel] / num_valid_neighbors;
	}



}

void fast_selective_blur_launch(cv::cuda::GpuMat& d_src, cv::cuda::GpuMat& d_dst, int threshold, int kernel_size, int width, int height) {



	dim3 threads_per_block(kernel_size, kernel_size, 3);
	dim3 num_blocks(width + 1, height + 1, 1);
	int substep_size = int(sizeof(uchar));
	int kernel_measure = (kernel_size - 1) / 2;

	selective_blur_kernel << < num_blocks, threads_per_block >> > (d_src.data, d_src.step, d_dst.data, threshold, width, height, substep_size, kernel_measure);


}










