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























bool Field::histograms_similar(cv::Mat hist_a, cv::Mat hist_b, int max_threshold, int sum_threshold) { //make it normalize histograms to account for differently sized regions
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

	if ((max_diff <= max_threshold)&&(sum_diff <= sum_threshold)) {
		similar = true;
	}

	return similar;
}

void Field::merge_regions(Region* region_keep, Region* region_clear) {

	//absorb = merge and set
	region_keep->absorb_constituents(region_clear->all_constituents()); 
	region_keep->absorb_histogram(region_clear->histogram()); 
	region_keep->absorb_neighbors(region_clear->all_neighboring_regions()); 

	if (region_clear->num_constituents() > 0) { region_clear->clear_constituents(); }
}

void Field::refine_region_sequence() {
	establish_region_neighbors(); 
	init_region_histograms(); 
	calculate_average_region_colors();

	connect_regions();
	refresh_regions();

	//prune_connections();
	//refine_regions();
	//refresh_regions();
}

void Field::establish_region_neighbors() {

	for (Region* focus : all_regions()) {
		std::set<Region*> new_neighbors;
		for (Superpixel* constituent : focus->all_constituents()) {
			for (Superpixel* neighbor : constituent->all_neighbors()) {
				new_neighbors.insert(neighbor->region());
			}
		}
		focus->set_neighboring_regions(new_neighbors);
	}
}

void Field::init_region_histograms() { //assumes each region only has one constituent. write a more general function later if you need it, but you probably wont.
	for (Region* region : all_regions()) {
		region->set_histogram(region->constituent_at(0)->histogram());
	}
}

//merges connected regions if the histograms are similar enough. start here.
void Field::connect_regions() {

	for (Region* focus : all_regions()) {
		if (focus->num_constituents() > 0) {
			for (Region* target : focus->all_neighboring_regions()) {
				if (target->num_constituents() > 0) {
					cv::Mat focus_hist = focus->histogram();
					cv::Mat target_hist = target->histogram();
					if (histograms_similar(focus_hist, target_hist, 180, 400)) { //make this function cooler
						merge_regions(focus, target);
					}
				}
			}
		}
	}
}

void Field::refresh_regions() { 
	std::set<Region*> new_regions;

	for (Region* region : all_regions()) {
		if (region->num_constituents() > 0) {
			new_regions.insert(region);
			std::set<Region*> new_neighbors;
			for (Region* neighbor : region->all_neighboring_regions()) {
				if (neighbor->num_constituents() > 0) {
					new_neighbors.insert(neighbor);
				}
			}
			region->set_neighboring_regions(new_neighbors);
		}
	}

	set_regions(new_regions);
}

void Field::refine_regions() {




}



















#pragma region refine regions naive

//if size of region is less than threshold, merge with the smallest nearby region. operates purely on size, not on content. maybe I can layer this in as an influencing factor later?
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
			merge_regions(smallest_neighbor, region); //smallest neighbor doesnt exist
		}
	}
}

//check if region only has one neighbor. if it does, merge with that neighbor. it didn't seem to work at all, but maybe it's just not working quite right. potentially worth revisiting.
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
	refresh_regions();
	calculate_average_region_colors();
	for (Region* region : all_regions()) {
		std::cout << "size: " << region->num_constituents() << std::endl;
	}
}

void Field::refine_region_sequence_naive() {
	connect_regions();
	calculate_average_region_colors();
	for (int i = 1; i <= 256; i *= 2) {
		prune_connections();
		refine_regions_naive(i);
		refresh_regions();
	}
}

#pragma endregion

void Field::connect_neighbors() {
	std::vector<thrust::pair<int, int>> pairs = GPU->find_borders(labels());

	for (thrust::pair<int, int> pair : pairs) {
		int first_label = pair.first;
		int second_label = pair.second;

		if (first_label != second_label) {
			Superpixel* focus = superpixel_at(first_label);
			Superpixel* target = superpixel_at(second_label);

			focus->add_neighbor(target);
		}
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

