#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"





Region::Region(Field* parent) {
	_parent = parent;
}

Region::~Region() {
}

cv::Point Region::id(){ return _constituents[0]->mean(); }

float Region::id_hash() {
	//biiig air quotes around "hash". only needs to be just good enough.
	cv::Point point_id = id();
	float elem_x = point_id.x;
	float elem_y = point_id.y;
	elem_x = (std::sqrt(elem_x) + 1) * 123;
	elem_y = (std::sqrt(elem_y) + 1) * 123;
	float hash = std::sqrt((elem_x * elem_y) + 1);
	return hash;
}


void Region::absorb_histogram(cv::Mat input_histogram) {

	cv::Mat combined_histogram(input_histogram.size(), input_histogram.type());


	cv::add(histogram(), input_histogram, combined_histogram);

	set_histogram(combined_histogram);
}

void Region::absorb_neighbors(std::set<Region*> input_neighbors) {

	std::set<Region*> output_neighbors;
	std::set<Region*> old_neighbors = all_neighboring_regions();
	
	std::merge(old_neighbors.begin(), old_neighbors.end(), input_neighbors.begin(), input_neighbors.end(), std::inserter(output_neighbors, output_neighbors.begin()));

	for (Region* region : output_neighbors) {
		region->add_neighboring_region(this);
	}

	set_neighboring_regions(output_neighbors);
}

void Region::absorb_constituents(std::vector<Superpixel*> input_constituents) {
	for (Superpixel* constituent : input_constituents) {
		constituent->set_region(this);
		add_constituent(constituent);
	}
}