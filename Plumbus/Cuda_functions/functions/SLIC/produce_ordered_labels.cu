#include"../../../cuda_includes.h"
#include"../../../config.h"




__global__ void raise_flags_kernel() {
	//for each pixel:
	//raise flag at label value
	//have: flags (1 by N)
}


__global__ void sum_flags_kernel() {
	//sum the values in flags to get k
	//probably better off writing and using exclusive sum general function here
}

__global__ void init_map_kernel() {
	//for each flag:
	//multiply flag by position, add one
}

//pop zeroes

__global__ void invert_map_kernel() {
	//for each value in map:
	//subtract one, insert the position of thread into index of map at the position equalling the value of the thread in orginal map
	//have: inverted_map (1 by N)
}

__global__ void assign_new_labels_kernel() {
	//for each pixel in labels:
	//get new value at index equaling label in inverted_maps
	//assign value as new pixel label
	//have: new_labels (w by h)
}






void produce_ordered_labels_launch(gMat& labels, int* num_labels) {
	int N = labels.rows * labels.cols;
	dim3 num_blocks_2d;
	dim3 threads_per_block_2d;
	boilerplate->get_kernel_structure(labels, &num_blocks_2d, &threads_per_block_2d, 2, 2);

	dim3 num_blocks_1d;
	dim3 threads_per_block_1d;
	boilerplate->get_kernel_structure(labels, &num_blocks_1d, &threads_per_block_1d, 2, 1);








}