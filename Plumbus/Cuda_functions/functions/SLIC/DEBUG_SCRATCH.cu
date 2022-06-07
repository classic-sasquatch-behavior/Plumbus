#include"../../../cuda_includes.h"
#include"../../../config.h"






//SEPARATE BLOBS - BROKEN
__global__ void linear_flow_kernel(iptr src, iptr temp, iptr id_LUT, int N, int* change) {
	get_dims_ids_and_check_bounds
		int self_label = src(row, col);
	int self_temp_val = temp(row, col);
	int greatest_temp_val = self_temp_val;

	for_each_immediate_neighbor(
		int neighbor_label = src(neighbor_row, neighbor_col);
	if (self_label == neighbor_label) {
		int neighbor_temp_val = temp(neighbor_row, neighbor_col);
		if (neighbor_temp_val > greatest_temp_val) {
			greatest_temp_val = neighbor_temp_val;
		}
	}
	) //end for_each_immediate_neighbor

		if (greatest_temp_val != self_temp_val) {
			change[0] = 1;
		}
}



__global__ void make_coords_to_ids_LUT_kernel(iptr src) {
	get_dims_ids_and_check_bounds
		src(row, col) = id;
}



void separate_blobs_launch(gMat& labels) {
	alias_input(labels);
	get_structure_from_mat;
	make_2d_kernel_from_structure;

	cv::cuda::GpuMat temp_labels = labels;
	cv::cuda::GpuMat id_LUT(labels.size(), labels.type());
	make_coords_to_ids_LUT_kernel << < num_blocks, threads_per_block >> > (id_LUT);
	cusyncerr(make_coords_to_ids_LUT_kernel);

	int change = 0;
	int* h_flag = &change;
	int* d_flag;
	cudaMalloc(&d_flag, sizeof(int));
	bool converged = false;
	int DEBUG_times_run = 0;

	while (!converged) {
		std::cout << "separate_blobs times run: " << DEBUG_times_run << std::endl;
		cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);
		linear_flow_kernel << <num_blocks, threads_per_block >> > (labels, temp_labels, id_LUT, N, d_flag);
		cusyncerr(linear_flow_in_separate_blobs);
		cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

		if (change == 0) {
			converged = true;
		}
		change = 0;
		DEBUG_times_run++;
	}
	cudaFree(d_flag);

	labels = temp_labels;
}
//END SEPARATE BLOBS





//ABSORB BLOBS - WORKING

__global__ void find_sizes_kernel(iptr src, iptr sizes) {
	get_dims_ids_and_check_bounds

		int self_label = src(row, col);
	atomicAdd(&sizes(0, self_label), 1);
}



__global__ void mark_labels_as_weak_kernel(iptr src, iptr sizes, iptr sizes_mat, iptr weakness, int threshold) {
	get_dims_ids_and_check_bounds

		int self_label = src(row, col);
	int size = sizes(0, self_label);

	if (size <= threshold) {
		weakness(row, col) = 1;
	}

	sizes_mat(row, col) = size;
}



__global__ void linear_flow_kernel(iptr src, iptr temp, iptr sizes_mat, iptr weakness, int* flag) {
	get_dims_ids_and_check_bounds

		int self_label = src(row, col);
	int self_size = temp(row, col);
	int greatest_size = self_size;
	int self_weakness = weakness(row, col);
	if (self_weakness == 1) {
		for_each_immediate_neighbor(
			int neighbor_size = temp(neighbor_row, neighbor_col);
		if (neighbor_size > greatest_size) {
			greatest_size = neighbor_size;
		}
		//might want to catch instances where sizes are the same. since its flowing though, this probably isnt actually a problem.
		) //end for_each_immediate_neighbor

			if (greatest_size != self_size) {
				flag[0] = 1;
				temp(row, col) = greatest_size;
			}
	}
}



void absorb_small_blobs_launch(gMat& input, int threshold) {
	get_structure_from_mat;
	make_2d_kernel_from_structure;

	cv::cuda::GpuMat sizes(cv::Size(N, 1), input.type());
	cv::cuda::GpuMat weakness(input.size(), input.type(), cv::Scalar{ 0 });
	cv::cuda::GpuMat sizes_mat(input.size(), input.type());

	find_sizes_kernel << <num_blocks, threads_per_block >> > (input, sizes);
	cusyncerr(find_sizes_in_absorb_small_blobs);
	mark_labels_as_weak_kernel << <num_blocks, threads_per_block >> > (input, sizes, sizes_mat, weakness, threshold);
	cusyncerr(mark_labels_as_weak);

	cv::cuda::GpuMat temp = sizes_mat;

	int change = 0;
	int* d_flag;
	int* h_flag = &change;
	cudaMalloc(&d_flag, sizeof(int));



	bool converged = false;
	while (!converged) {

		cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);
		linear_flow_kernel << <num_blocks, threads_per_block >> > (input, temp, sizes_mat, weakness, d_flag);
		cusyncerr(linear_flow_in_absorb_small_blobs);
		cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

		if (change == 0) {
			converged = true;
		}
		change = 0;

	}

	cudaFree(d_flag);
}
//END ABSORB BLOBS


//SEPARATE BLOBS CLOSER LOOK AT FLOW
__global__ void linear_flow_kernel(iptr src, iptr temp, iptr id_LUT, int N, int* change) {
	get_dims_ids_and_check_bounds
		int self_label = src(row, col);
	int self_temp_val = temp(row, col);
	int greatest_temp_val = self_temp_val;

	for_each_immediate_neighbor(
		int neighbor_label = src(neighbor_row, neighbor_col);
	if (self_label == neighbor_label) {
		int neighbor_temp_val = temp(neighbor_row, neighbor_col);
		if (neighbor_temp_val > greatest_temp_val) {
			greatest_temp_val = neighbor_temp_val;
		}
	}
	) //end for_each_immediate_neighbor

		if (greatest_temp_val != self_temp_val) {
			change[0] = 1;
		}
}
//END SEPARATE BLOBS FLOW

//ABSORB BLOBS CLOSER LOOK AT FLOW

__global__ void linear_flow_kernel(iptr src, iptr temp, iptr sizes_mat, iptr weakness, int* flag) {
	get_dims_ids_and_check_bounds

		int self_label = src(row, col);
	int self_size = temp(row, col);
	int greatest_size = self_size;
	int self_weakness = weakness(row, col);
	if (self_weakness == 1) {
		for_each_immediate_neighbor(
			int neighbor_size = temp(neighbor_row, neighbor_col);
		if (neighbor_size > greatest_size) {
			greatest_size = neighbor_size;
		}
		//might want to catch instances where sizes are the same. since its flowing though, this probably isnt actually a problem.
		) //end for_each_immediate_neighbor

			if (greatest_size != self_size) {
				flag[0] = 1;
				temp(row, col) = greatest_size;
			}
	}
}



//END ABSORB BLOBS FLOW