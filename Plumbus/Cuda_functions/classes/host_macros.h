#pragma once





#define for_row_col(max_rows, max_cols, content)   \
for(int row = 0; row < max_rows; row++){			\
	for (int col = 0; col < max_cols; col++) {		\
		content										\
	}												\
}