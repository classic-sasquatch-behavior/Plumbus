#include"../../cuda_includes.h"








__global__ void find_borders_kernel() {

}








std::vector<int[2]> find_borders_launch() {
	dim3 num_blocks;
	dim3 threads_per_block;




	find_borders_kernel << < num_blocks, threads_per_block >> > ();
}