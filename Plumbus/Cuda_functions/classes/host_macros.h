#pragma once





#define for_row_col(max_rows, max_cols, content)   \
for(int row = 0; row < max_rows; row++){			\
	for (int col = 0; col < max_cols; col++) {		\
		content										\
	}												\
}



#define for_i_j(max_i, max_j, content)   \
for(int i = 0; i < max_i; i++){			\
	for (int j = 0; j < max_j; j++) {		\
		content										\
	}												\
}