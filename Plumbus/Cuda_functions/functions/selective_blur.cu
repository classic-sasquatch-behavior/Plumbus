#include"../../cuda_includes.h"










__global__ void selective_blur_kernel(uchar* src, int step, uchar* dst, int threshold, int kernel_min, int kernel_max, int width, int height, int substep) {

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int row = yIndex;
	const int col = xIndex;
	
	if ((col >= width) || (row >= height)) { return; }

	int focus_address = (row * step) + (col * substep * 3);
	int num_valid_neighbors = 0;
	int channel_avg[3] = { 0,0,0 };
	int focus_color[3] = {src[focus_address], src[focus_address + substep], src[focus_address + (2*substep)]};

	for (int irow = kernel_min; irow <= kernel_max; irow++) {
		for (int icol = kernel_min; icol <= kernel_max; icol++) {
			int target_row = row + irow;
			int target_col = col + icol;
			if ((target_row > -1)&&(target_col > -1)&&(target_row < height)&&(target_col < width)) {
				int target_address = (target_row * step)+(target_col * substep * 3);

				int total_difference = 0;
				int target_color[3] = { src[target_address], src[target_address + substep], src[target_address + (2*substep)] };
				for (int channel = 0; channel < 3; channel++) {
					total_difference = total_difference + (abs(focus_color[channel] - target_color[channel]));
				}

				//problem must be below here - pixels seem to turn black only when they are near an identical pixel
				if (total_difference <= threshold) {
					num_valid_neighbors = num_valid_neighbors + 1;
					for (int channel = 0; channel < 3; channel++) {
						channel_avg[channel] = channel_avg[channel] + target_color[channel];
					}
				}
			}
		}
	}


	dst[focus_address] = channel_avg[0] / num_valid_neighbors;
	dst[focus_address + substep] = channel_avg[1] / num_valid_neighbors;
	dst[focus_address + (2*substep)] = channel_avg[2] / num_valid_neighbors;

}

void selective_blur_launch(cv::cuda::GpuMat &d_src, cv::cuda::GpuMat &d_dst, int threshold, int kernel_min, int kernel_max, int width, int height) {

	dim3 threads_per_block(32, 32, 1);
	dim3 num_blocks((width / 32) + 1, (height / 32) + 1, 1);
	int substep_size = int(sizeof(uchar));

	selective_blur_kernel <<< num_blocks, threads_per_block >>> (d_src.data, d_src.step, d_dst.data, threshold, kernel_min, kernel_max, width, height, substep_size);


}










