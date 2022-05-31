#include"../../cuda_includes.h"






void update_centers_launch(gMat& labels, gMat& row_vals, gMat& col_vals, int* average_displacement, int K_rows, int K_cols, int K);