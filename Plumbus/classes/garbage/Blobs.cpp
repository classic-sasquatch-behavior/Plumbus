#include"../../includes.h"
#include"../../classes.h"


const int LAYER_WIDTH = 16;

//-----objects-----//

ProtoObject::ProtoObject() {

}

//-----edges-----//



//-----blobs-----//

#pragma region Blobs

BlobImage::BlobImage(cv::Mat source) {
	_source = source;
}

BlobPlane::BlobPlane(std::vector<cv::Mat> source, int plane_index) {
	_source = source;
	_plane_index = plane_index;
}

BlobLayer::BlobLayer(cv::Mat source, BlobPlane* parent, int layer_index) {
	_source = source;
	_parent = parent;
	_layer_index = layer_index;
}

void BlobLayer::initialize_valid_pixel_map() {
	std::vector<std::vector<BlobPixel*>> result;

	for (int row = 0; row < _source.rows; row++) {
		std::vector<BlobPixel*> new_col;
		for (int col = 0; col < _source.cols; col++) {
			new_col.push_back(nullptr);
		}
		result.push_back(new_col);
	}

	for (BlobPixel* pixel : _valid_pixels) {
		result[pixel->row()][pixel->col()] = pixel;
	}

	_valid_pixel_map = result;
}

BlobPixel* BlobLayer::get_pixel_by_coordinates(int row, int col) {
	return _valid_pixel_map[row][col];
}

Blob::Blob(BlobLayer* parent) {
	_parent = parent;
	_layer_index = parent->layer_index();
	cv::Vec3b base_color = { 75,75,75 };
	int plane_index = _parent->parent()->plane_index();
	base_color[plane_index] = (_layer_index * LAYER_WIDTH) + (LAYER_WIDTH / 2);
	_color = base_color;
}

SelfPixel::SelfPixel(int relative_row, int relative_col, Blob* parent, int absolute_row, int absolute_col) {
	_absolute_row = absolute_row;
	_absolute_col = absolute_col;
	_parent = parent;
	_relative_row = relative_row;
	_relative_col = relative_col;
}

Border::Border(int relative_row, int relative_col, Blob* parent, int absolute_row, int absolute_col) :
	SelfPixel(relative_row, relative_col, parent, absolute_row, absolute_col) {


}

void Blob::conceive_self() {

	std::vector<std::vector<SelfPixel*>> temp_self;
	_row_width = _highest_col - _lowest_col; //should be working correctly, but if you get an out of bounds error the ordering here is whats causing it
	_col_height = _highest_row - _lowest_row; //see above


	for (int row = 0; row < _col_height + 2; row++) {
		std::vector<SelfPixel*> new_row;
		for (int col = 0; col < _row_width + 2; col++) {
			new_row.push_back(nullptr); //change to puyshing back space later if you feel like that's necessary. we don't need space beyond edges though, at least not right now.
		}
		temp_self.push_back(new_row);
	}

	for (BlobPixel* constituent : _constituents) {
		int absolute_row = constituent->row();
		int absolute_col = constituent->col();
		int relative_row = absolute_row - _anchor.first; //is this correct?
		int relative_col = absolute_col - _anchor.second; //same here, is this correct? I am trying to get unbiased coordinates
		SelfPixel* new_pixel = new SelfPixel(relative_row, relative_col, this, absolute_row, absolute_col);
		temp_self[relative_row][relative_col] = new_pixel;
		_self_pixels.push_back(new_pixel);
	} //so now we have a "matrix" (v of v) of nullptr and pixels.

	_conception_of_self = temp_self;

}

void Blob::determine_borders() { //this is slow sometimes when the others pretty much arent, theres probably something wrong with it
	timer->begin("determine borders");

	std::queue<std::pair<int, int>> search_queue;
	std::set<std::pair<int, int>> already_searched;

	int initial_row = _self_pixels[0]->relative_row();
	int initial_col = _self_pixels[0]->relative_col();
	std::pair<int, int> initial_coordinates = { initial_row, initial_col };
	search_queue.push(initial_coordinates);

	while (!search_queue.empty()) {
		std::pair<int, int> target_coordinates = search_queue.front();
		search_queue.pop();
		//std::cout << target_coordinates.first << ", " << target_coordinates.second << std::endl;
		if (already_searched.find(target_coordinates) == already_searched.end()) {
			already_searched.insert(target_coordinates);
			for (int row = -1; row < 2; row++) {
				for (int col = -1; col < 2; col++) {
					int inner_target_relative_row = target_coordinates.first + row;
					int inner_target_relative_col = target_coordinates.second + col;
					std::pair<int, int> inner_target_relative_coordinates = { inner_target_relative_row, inner_target_relative_col };

					if (((inner_target_relative_row >= 0) && (inner_target_relative_row < _conception_of_self.size()))
						&& ((inner_target_relative_col >= 0) && (inner_target_relative_col < _conception_of_self[0].size()))) {

						if (already_searched.find(inner_target_relative_coordinates) == already_searched.end()) {
							SelfPixel* inner_target = _conception_of_self[inner_target_relative_row][inner_target_relative_col];

							if (inner_target == nullptr) {
								//std::wcout << "border detected" << std::endl;
								int border_rel_row = inner_target_relative_row;
								int border_rel_col = inner_target_relative_col;
								int border_abs_row = inner_target_relative_row + _anchor.first;
								int border_abs_col = inner_target_relative_col + _anchor.second;
								Border* border = new Border(border_rel_row, border_rel_col, this, border_abs_row, border_abs_col);
								_conception_of_self[inner_target_relative_row][inner_target_relative_col] = border;
								_border_pixels.push_back(border);
								already_searched.insert(inner_target_relative_coordinates);
							}

							else {
								//push back cell to queue
								//already_searched.insert(inner_target_relative_coordinates);

								search_queue.push(inner_target_relative_coordinates);
							}
						}
					}
				}
			}
		}
	}
	timer->end("determine borders");
}

void Blob::fill_border_pixels() {
	for (Border* border_pixel : _border_pixels) {
		//this is where you need the image map. or actually, for now the planar map.
		int abs_row = border_pixel->absolute_row();
		int abs_col = border_pixel->absolute_col();
		if (((abs_row < 0) || (abs_row >= _parent->source().rows)) || ((abs_col < 0) || (abs_col >= _parent->source().cols))) {
			continue;
		}
		else {
			Blob* border_parent = _parent->parent()->get_occupant(abs_row, abs_col);
			border_pixel->set_parent(border_parent);
			_border_blobs.insert(border_parent);
		}
	}
}

void Blob::associate_blobs() {
	for (Blob* border_blob : _border_blobs) {
		int border_blob_layer = border_blob->parent()->layer_index();
		int this_layer = parent()->layer_index();
		int distance = abs(border_blob_layer - this_layer);
		_associations.push_back({ border_blob, distance });
	}
}

void BlobPlane::draw_map() {
	for (int row = 0; row < _source[0].rows; row++) {
		std::vector<Blob*> new_row;
		for (int col = 0; col < _source[0].cols; col++) {
			new_row.push_back(nullptr);
		}
		_map.push_back(new_row);
	}

	for (Blob* blob : _blobs) {
		for (BlobPixel* pixel : blob->all_constituents()) {
			int row = pixel->row();
			int col = pixel->col();
			_map[row][col] = blob;
		}
	}
}

void Blob::add_constituent(BlobPixel* input) {
	_constituents.push_back(input);
	int row_in = input->row();
	int col_in = input->col();

	if (row_in > _highest_row) {
		_highest_row = row_in;
	}
	if (row_in < _lowest_row) {
		_lowest_row = row_in;
	}
	if (col_in > _highest_col) {
		_highest_col = col_in;
	}
	if (col_in < _lowest_col) {
		_lowest_col = col_in;
	}

}

BlobPixel::BlobPixel(int row, int col) {
	_row = row;
	_col = col;
}
#pragma endregion















