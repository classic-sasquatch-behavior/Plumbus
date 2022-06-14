#pragma once
#include"../../../cuda_includes.h"




void assign_pixels_to_centers_launch(gMat& src_L, gMat& src_A, gMat& src_b, gMat& labels, gMat& row_vals, gMat& col_vals, int sector_rows, int sector_cols, int S, int dmod);
