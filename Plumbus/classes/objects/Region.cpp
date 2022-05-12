#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"





Region::Region(Field* parent) {
	_parent = parent;
}

Region::~Region() {
}

cv::Point Region::id(){ return _constituents[0]->mean(); }


void Region::absorb_histogram(cv::Mat input_histogram) {

	cv::Mat combined_histogram(input_histogram.size(), input_histogram.type());

	std::vector<cv::Mat> histograms_to_merge = { histogram(), input_histogram };

	cv::merge(histograms_to_merge, combined_histogram);

	set_histogram(combined_histogram);
}

void Region::absorb_neighbors(std::set<Region*> input_neighbors) {

	std::set<Region*> output_neighbors;
	std::set<Region*> old_neighbors = all_neighboring_regions();
	
	std::merge(old_neighbors.begin(), old_neighbors.end(), input_neighbors.begin(), input_neighbors.end(), std::inserter(output_neighbors, output_neighbors.begin()));

	set_neighboring_regions(output_neighbors);
}

void Region::absorb_constituents(std::vector<Superpixel*> input_constituents) {
	for (Superpixel* constituent : input_constituents) {
		constituent->set_region(this);
		add_constituent(constituent);
	}
}