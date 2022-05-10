#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"

Field::Field(Frame* frame, cv::Mat labels) {
	_frame = frame;
	_labels = labels;
}

Field::~Field() {

}

void Field::calculate_average_region_colors() {
	for (Region* region : all_regions()) {
		std::vector<int> color_sum = { 0,0,0 };

		for (Superpixel* constituent : region->all_constituents()) {
			cv::Vec3b constituent_color = constituent->average_color();
			for (int channel = 0; channel < 3; channel++) {
				color_sum[channel] += constituent_color[channel];
			}
		}

		cv::Vec3b color_out;
		for (int channel = 0; channel < 3; channel++) {
			color_out[channel] = color_sum[channel] / region->num_constituents();
		}
		/*region->set_average_color(color_out);*/
		region->set_average_color(colorwheel->get_color(32));
	}
}


bool Field::histograms_similar(cv::Mat hist_a, cv::Mat hist_b, int max_threshold, int sum_threshold) {
	bool similar = false;
	cv::Mat hist_diff(hist_a.size(), hist_a.type());
	cv::absdiff(hist_a, hist_b, hist_diff);

	std::vector<cv::Mat> channels;
	cv::split(hist_diff, channels);

	cv::Scalar sum_diff_scalar = cv::sum(hist_diff);
	float sum_diff = sum_diff_scalar[0] + sum_diff_scalar[1] + sum_diff_scalar[2];

	double min_diff;
	double max_diff;
	cv::minMaxIdx(hist_diff, &min_diff, &max_diff);

	//std::cout << "max diff example: " << max_diff << std::endl;
	//std::cout << "sum diff example: " << sum_diff << std::endl;
	//std::cout << std::endl;

	if ((max_diff <= max_threshold)&&(sum_diff <= sum_threshold)) {
		similar = true;
	}

	return similar;
}

void Field::merge_regions(Region* region_keep, Region* region_clear) {

	for (Superpixel* constituent : region_clear->all_constituents()) {
		constituent->set_region(region_keep);
		region_keep->add_constituent(constituent);
	}
	if (region_clear->num_constituents() > 0) { region_clear->clear_constituents(); }
}

void Field::connect_regions() { //sucks real bad. too long and too much memory usage
	std::set<Region*> new_regions;

	for (Superpixel* focus : all_superpixels()) {
		Region* new_region = focus->region();
		//std::cout << "new focus" << std::endl;
		for (Superpixel* target : focus->all_neighbors()) {
			if (target->region()->all_constituents()[0]->mean() != focus->region()->all_constituents()[0]->mean()) {

				cv::Mat target_hist = target->histogram();
				cv::Mat focus_hist = focus->histogram();
				if (histograms_similar(target_hist, focus_hist, 180, 400)) { //80, 340 looking good on source //180, 450 looking really good for blur 
					//std::cout << "regions merged" << std::endl;

					Region* dead_region = target->region();
					for (Superpixel* constituent : dead_region->all_constituents()) {
						constituent->set_region(new_region);
						new_region->add_constituent(constituent);
					}
					if(dead_region->num_constituents() > 0){ dead_region->clear_constituents(); }
					//delete dead_region;
				}
			}

		}
		new_regions.insert(new_region);
	}

	std::set<Region*> newer_regions;
	for (Region* region : new_regions) {
		if (region->num_constituents() > 0) {
			newer_regions.insert(region);
		}
	}
	set_regions(newer_regions);
}

void Field::prune_connections() {

	for (Superpixel* focus : all_superpixels()) {
		std::vector<Superpixel*> new_neighbors;

		cv::Point focus_id = focus->region()->id();
		for (Superpixel* target : focus->all_neighbors()) {
			cv::Point target_id = target->region()->id();

			if (focus_id != target_id) {
				new_neighbors.push_back(target);
			}
		}
		focus->set_neighbors(new_neighbors);
	}
}

void Field::refine_region_sequence() {
	prune_connections();
	refine_regions();
	refresh_region_list();
}

void Field::refine_regions() {





















}

#pragma region refine regions wip

void Field::refine_region_sequence_naive() {
	for (int i = 1; i <= 8; i *= 2) {
		prune_connections();
		refine_regions_naive(i);
		refresh_region_list();
	}
}

void Field::refine_regions_naive(int threshold) {

	for (Region* region : all_regions()) {
		if (region->num_constituents() <= threshold && region->num_constituents() > 0) {

			cv::Point focus_region_id = region->id();
			Region* smallest_neighbor = nullptr;
			int size_of_smallest_neighbor = INF;

			for (Superpixel* constituent : region->all_constituents()) {
				for (Superpixel* neighbor : constituent->all_neighbors()) {
					cv::Point neighbor_region_id = neighbor->region()->id();


					if (neighbor_region_id != focus_region_id) {
						int size_of_this_neighbor = neighbor->region()->num_constituents();
						if (size_of_this_neighbor < size_of_smallest_neighbor && size_of_this_neighbor != 0) {

							smallest_neighbor = neighbor->region();
							size_of_smallest_neighbor = size_of_this_neighbor;
						}
					}
				}
			}

			merge_regions(smallest_neighbor, region);


		
			
		}
	}


}

void Field::refine_regions_old() {

	for (Region* region : all_regions()) {
		if (region->num_constituents() != 0) {

			cv::Point region_id = region->id();
			cv::Point solo_id;
			Region* merge_with = nullptr;
			bool found_neighbor = false;
			bool region_invalid = false;

			for (Superpixel* constituent : region->all_constituents()) {

				for (Superpixel* neighbor : constituent->all_neighbors()) {
					cv::Point neighbor_id = neighbor->region()->id();
					if (neighbor_id != solo_id && neighbor_id != region_id) {
						if (!found_neighbor) {
							found_neighbor = true;
							solo_id = neighbor_id;
							merge_with = neighbor->region();
						}
						else {
							region_invalid = true;
							break;
						}
					} //do I want to break here?
				}

				if (region_invalid) {
					break;
				}
			}

			if (!region_invalid) {
				std::cout << "refining regions" << std::endl;
				merge_regions(region, merge_with);
				//merge_with->clear_constituents();
			}
		}
	}
	refresh_region_list();
	calculate_average_region_colors();
	for (Region* region : all_regions()) {
		std::cout << "size: " << region->num_constituents() << std::endl;
	}
}

#pragma endregion

void Field::refresh_region_list() {
	std::set<Region*> new_regions;

	for (Region* region : all_regions()) {
		if (region->num_constituents() > 0) {
			new_regions.insert(region);
		}
	}

	set_regions(new_regions);
}

void Field::connect_neighbors() {
	const int NEIGHBOR_KERNEL_SIZE = 5;
	const int KERNEL_MAX = (NEIGHBOR_KERNEL_SIZE - 1) / 2;
	const int KERNEL_MIN = -1 * KERNEL_MAX;

	for (int row = 0; row < bin_rows(); row++) {
		for (int col = 0; col < bin_cols(); col++) {
			std::vector<Superpixel*> target_bin = bin_at(row, col);

			for (Superpixel* superpixel: target_bin) {
				for (int irow = KERNEL_MIN; irow <= KERNEL_MAX; irow++) {
					for (int icol = KERNEL_MIN; icol <= KERNEL_MAX; icol++) {
						int target_row = row + irow;
						int target_col = col + icol;
						if ((target_row > -1)&&(target_col > -1)&&(target_row < bin_rows())&&(target_col < bin_cols())) {
							std::vector<Superpixel*> neighbor_bin = bin_at(target_row, target_col);
							for (Superpixel* target : neighbor_bin) {
								if (target->mean() != superpixel->mean()) {
									superpixel->add_neighbor(target);
								}
							}
						}
					}
				}
			}
		}
	}
}

void Field::initialize_bins() {
	int original_num_pixels = original_rows() * original_cols();

	int area_ratio = original_num_pixels / num_superpixels();

	float actual_shrinkage = sqrt(area_ratio);

	int new_rows = std::ceil(original_rows() / std::floor(actual_shrinkage));
	int new_cols = std::ceil(original_cols() / std::floor(actual_shrinkage));


	for (int row = 0; row < new_rows; row++) {
		std::vector<std::vector<Superpixel*>> new_row;
		for (int col = 0; col < new_cols; col++) {
			std::vector<Superpixel*> new_bin;
			new_row.push_back(new_bin);
		}
		_bins.push_back(new_row);
	}

	_bin_size = std::ceil(original_rows() / new_rows);
	
	for (int i = 0; i < num_superpixels(); i++) {
		Superpixel* target = superpixel_at(i);
		int target_row = target->mean().y;
		int target_col = target->mean().x;

		int bin_row = std::floor(target_row / bin_size());
		int bin_col = std::floor(target_col / bin_size());

		add_to_bin_at(std::min(bin_row, new_rows - 1), std::min(bin_col, new_cols - 1), target);
	}

}

void Field::form_regions() {
	int index = 0;
	for (Superpixel* superpixel : all_superpixels()) {
		Region* new_region = new Region(this);
		superpixel->set_region(new_region);
		new_region->add_constituent(superpixel);
		add_region(new_region);
		index++;
	}
}

