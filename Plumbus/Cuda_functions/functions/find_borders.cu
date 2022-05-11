#include"../../cuda_includes.h"








__global__ void find_borders_kernel(int width, int height, uchar* d_src, int src_row_step, int* d_focus, int* d_target, int src_elm_size) { 

	int focus_x = blockIdx.x;
	int focus_y = blockIdx.y;

	int target_x_mod = (threadIdx.x % 3) - 1;
	int target_y_mod = ((threadIdx.x - target_x_mod) / 3) - 1;

	int target_x = focus_x + target_x_mod;
	int target_y = focus_y + target_y_mod;

	if (focus_x >= width || focus_y >= height || target_x >= width || target_y >= height || target_x < 0 || target_y < 0) { return; }

	int write_position = (((blockIdx.x + (gridDim.x * blockIdx.y)) * 9) + threadIdx.x); //position within output array, which is linear
	int focus_position = (focus_y * src_row_step)+(focus_x * src_elm_size); //position within input mat, which is width*height
	int target_position = (target_y * src_row_step)+(target_x * src_elm_size); //position within input mat, which is width*height

	int focus_val = d_src[focus_position];
	int target_val = d_src[target_position];

	if (focus_val != target_val) {
		d_focus[write_position] = focus_val;
		d_target[write_position] = target_val;
	}
	else {
		d_focus[write_position] = -1;
		d_target[write_position] = -1;
	}
}






struct make_pair : public thrust::binary_function<int, int, thrust::pair<int, int>> {
	__host__ __device__
		thrust::pair<int, int> operator()(int x, int y) { return thrust::make_pair(x, y); }
};


std::vector<thrust::pair<int, int>> find_borders_launch(int src_width, int src_height, cv::cuda::GpuMat& d_src) { //just make it return a thrust::host_vector

	thrust::device_vector<int> d_focus_result(src_width * src_height * 9);
	thrust::device_vector<int> d_target_result(src_width * src_height * 9);

	int* d_focus = thrust::raw_pointer_cast(d_focus_result.data());
	int* d_target = thrust::raw_pointer_cast(d_target_result.data());

	dim3 num_blocks = {uint(src_width + 1), uint(src_height + 1)};
	dim3 threads_per_block = {9, 1, 1};
	int substep_size = sizeof(int);

	find_borders_kernel <<<num_blocks, threads_per_block>>> (src_width, src_height, d_src.data, d_src.step, d_focus, d_target, substep_size);


	//SOLVED: USE TUPLES AND A ZIP ITERATOR, NOT PAIR AND WHATEVER THIS SHIT IS



	thrust::device_vector<thrust::pair<int, int>> d_unique(d_focus_result.size());


	thrust::transform(d_focus_result.begin(), d_focus_result.end(), d_target_result.begin(), d_unique.begin(), make_pair() ); //seems to do its job just fine

	thrust::unique(thrust::device, d_unique.begin(), d_unique.end());

	std::vector<thrust::pair<int, int>> output(d_unique.size());
	thrust::copy(d_unique.begin(), d_unique.end(), output.begin() );
	return output;
}