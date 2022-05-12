#pragma once
#include"../../includes.h"

class Field;
class Superpixel;


class Region {
public:
	Region(Field* parent);
	~Region();
	inline void clear_constituents() { _constituents.clear(); }
	void absorb_histogram(cv::Mat input_histogram);
	void absorb_neighbors(std::set<Region*> input_neighbors);
	void absorb_constituents(std::vector<Superpixel*> input_constituents);

#pragma region get-set

	inline std::vector<Superpixel*> all_constituents() { return _constituents; }
	inline Field* parent() { return _parent; }
	inline cv::Vec3b average_color() { return _average_color; }
	cv::Point id();
	cv::Mat histogram() { return _histogram; }
	std::set<Region*> all_neighboring_regions() { return _neighboring_regions; }
	Superpixel* constituent_at(int index) { return _constituents[index]; }

	void set_neighboring_regions(std::set<Region*> input) { _neighboring_regions = input; }
	void add_neighboring_region(Region* input) { _neighboring_regions.insert(input); }
	void set_histogram(cv::Mat input) { _histogram = input; }
	inline void set_average_color(cv::Vec3b input) { _average_color = input; }
	inline void add_constituent(Superpixel* input) { _constituents.push_back(input); }
	inline void set_parent(Field* input) { _parent = input; }

	inline int num_constituents() { return _constituents.size(); }
#pragma endregion





private:
	std::vector<Superpixel*> _constituents;
	Field* _parent;
	cv::Vec3b _average_color;
	cv::Mat _histogram;

	std::set<Region*> _neighboring_regions;
};