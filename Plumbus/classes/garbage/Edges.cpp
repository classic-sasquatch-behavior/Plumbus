#include"../../includes.h"
#include"../../classes.h"







#pragma region Edges

Edge::Edge() {
	_max_row = -1;
	_max_col = -1;
	_min_row = 200000;
	_min_col = 200000;
}

void Edge::addConstituent(EdgePixel* input) {
	_constituents.push_back(input);
	int absolute_row = input->row();
	int absolute_col = input->col();

	if (absolute_row < _min_row) {
		_min_row = absolute_row;
	}
	if (absolute_col < _min_col) {
		_min_col = absolute_col;
	}
	if (absolute_row > _max_row) {
		_max_row = absolute_row;
	}
	if (absolute_col > _max_col) {
		_max_col = absolute_col;
	}

}

//edge pixels//

EdgePixel::EdgePixel(int row, int col) {
	_row = row;
	_col = col;
}

//edge image//

EdgeImage::EdgeImage() {}

void EdgeImage::addPixel(EdgePixel* input) {

	_valid_pixels.push_back(input);







}

void EdgeImage::initialize_valid_pixel_map() {
	std::vector<std::vector<EdgePixel*>> result;

	for (int row = 0; row < _num_rows; row++) {
		std::vector<EdgePixel*> new_col;
		for (int col = 0; col < _num_cols; col++) {
			new_col.push_back(nullptr);
		}
		result.push_back(new_col);
	}

	for (EdgePixel* pixel : _valid_pixels) {
		result[pixel->row()][pixel->col()] = pixel;
	}

	_valid_pixel_map = result;
}

bool EdgeImage::neighbor_exists(int row, int col) {
	for (EdgePixel* pixel : _valid_pixels) {
		if ((pixel->row() == row) && (pixel->col() == col)) {
			return true;
		}
	}
	return false;
}

EdgePixel* EdgeImage::get_pixel(int row, int col) {
	EdgePixel* target = nullptr;
	for (EdgePixel* pixel : _valid_pixels) {
		if ((pixel->row() == row) && (pixel->col() == col)) {
			target = pixel;
			break;
		}
	}
	return target;
}

EdgePixel* EdgeImage::get_pixel_by_coordinates(int row, int col) {
	return _valid_pixel_map[row][col];
}

#pragma endregion






















