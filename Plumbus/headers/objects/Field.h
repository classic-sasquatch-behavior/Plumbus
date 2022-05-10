#pragma once
#include"../../includes.h"

class Frame;
class Superpixel;
class Region;




class Field {
public:
	Field(Frame* frame, cv::Mat labels);
	~Field();
	void connect_neighbors();
	void form_regions();
	void initialize_bins();
	void connect_regions();
	bool histograms_similar(cv::Mat hist_a, cv::Mat hist_b, int max_threshold, int sum_threshold);
	void calculate_average_region_colors();
	void prune_connections();
	void refine_region_sequence_naive();
	void refine_region_sequence();
	void refine_regions();
	void refine_regions_naive(int threshold);
	void refine_regions_old();
	void merge_regions(Region* region_keep, Region* region_clear);
	void refresh_region_list();

#pragma region get-set
	inline Frame* frame() { return _frame; }
	inline std::vector<Superpixel*> all_superpixels() { return _superpixels; }
	inline Superpixel* superpixel_at(int index) { return _superpixels[index]; }
	inline cv::Mat labels() { return _labels; }
	inline std::vector<cv::Point> all_means() { return _means; }
	inline cv::Point mean_at(int index) { return _means[index]; }
	inline std::vector<Superpixel*> bin_at(int row, int col) { return _bins[row][col]; }
	inline int original_rows() { return _labels.rows; }
	inline int original_cols() { return _labels.cols; }
	inline int bin_rows() { return _bins.size(); }
	inline int bin_cols() { return _bins[0].size(); }
	inline int bin_size() { return _bin_size; }
	inline std::set<Region*> all_regions() { return _regions; }

	inline void set_regions(std::set<Region*> input) { _regions = input; }
	inline void add_region(Region* input) { _regions.insert(input); }
	inline void add_to_bin_at(int row, int col, Superpixel* input) { _bins[row][col].push_back(input); }
	inline void add_mean(cv::Point input) { _means.push_back(input); }
	inline void set_frame(Frame* input) { _frame = input; }
	inline void set_labels(cv::Mat input) { _labels = input; }
	inline void add_superpixel(Superpixel* input) { _superpixels.push_back(input); }
	inline void set_superpixels(std::vector<Superpixel*> input) { _superpixels = input; }

	inline int num_superpixels() { return _superpixels.size(); }
	inline int num_regions() { return _regions.size(); }
#pragma endregion

private:
	Frame* _frame;
	std::vector<Superpixel*> _superpixels;
	std::vector<cv::Point> _means;
	cv::Mat _labels;
	std::vector<std::vector<std::vector<Superpixel*>>> _bins;
	int _bin_size;
	std::set<Region*> _regions;
};