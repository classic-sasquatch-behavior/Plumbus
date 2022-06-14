#include"../../../cuda_includes.h"
#include"../../../config.h"







__global__ void assign_pixels_to_centers_kernel( iptr labels, iptr src_L, iptr src_A, iptr src_B, iptr row_vals, iptr col_vals, int sector_rows, int sector_cols, int S, int dmod) {
	alias_src(labels);
	get_dims_ids_and_check_bounds;

	int sector_row = floorf(row/S);
	int sector_col = floorf(col/S);
	int sector_id = (sector_row * sector_cols) + sector_col;

	int neighbors[3][3] = { -1 };

	for_each_immediate_neighbor_specific(sector_rows, sector_cols, sector_row, sector_col,
		int neighbor_id = (neighbor_row * sector_cols) + neighbor_col;
		neighbors[neighbor_row][neighbor_col] = neighbor_id;
	);

	int self_l = src_L(row, col);
	int self_a = src_A(row, col);
	int self_b = src_B(row, col);

	int closest_neighbor = -1;
	float closest_neighbor_distance = INFINITY;


	for_i_j(3, 3,
		int target_center_id = neighbors[i][j];
		if (target_center_id == -1) { break; }
		int target_center_actual_row = row_vals(0, target_center_id);
		int target_center_actual_col = col_vals(0, target_center_id);

		int target_l = src_L(target_center_actual_row, target_center_actual_col);
		int target_a = src_A(target_center_actual_row, target_center_actual_col);
		int target_b = src_B(target_center_actual_row, target_center_actual_col);

		float dxy = sqrtf(((target_center_actual_row - row)*(target_center_actual_row - row))+ ((target_center_actual_col - col) * (target_center_actual_col - col)));
		float dlab = sqrtf(((self_l - target_l)*(self_l*target_l))+((self_a - target_a)*(self_a - target_a))+((self_b - target_b)*(self_b - target_b)));
		float D = dlab + (dmod*dxy);

		if (D < closest_neighbor_distance) {
			closest_neighbor_distance = D;
			closest_neighbor = target_center_id;
		}
	);

	labels(row, col) = closest_neighbor;

}











void assign_pixels_to_centers_launch(gMat &src_L, gMat & src_A, gMat & src_B, gMat& labels, gMat& row_vals, gMat& col_vals, int sector_rows, int sector_cols, int S, int dmod){
	alias_input(labels);
	get_structure_from_mat;
	make_2d_kernel_from_structure;

	assign_pixels_to_centers_kernel<<<num_blocks, threads_per_block>>>(src_L, src_A, src_B, labels, row_vals, col_vals, sector_rows, sector_cols, S, dmod);
	cusyncerr(assign_pixels_to_centers);
}