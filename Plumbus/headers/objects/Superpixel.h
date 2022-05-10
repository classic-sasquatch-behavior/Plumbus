#pragma once
#include"../../includes.h"


class Field;


class Superpixel {
public:
	Superpixel(Field* parent);
	~Superpixel();
	void compute_average_color();
	void compute_mean_of_points();
	void compute_histogram();

#pragma region get-set
	inline cv::Point mean() { return _mean; }
	inline Field* parent() { return _parent; }
	inline std::vector<cv::Point> all_points() { return _points; }
	inline std::vector<cv::Vec3b> all_colors() { return _colors; }
	inline cv::Point point_at(int index) { return _points[index]; }
	inline cv::Vec3b average_color() { return _average_color; }
	inline cv::Vec3b color_at(int index) { return _colors[index]; }
	inline cv::Mat histogram() { return _histogram; }
	inline std::vector<Superpixel*> all_neighbors() { return _neighbors; }
	inline Region* region() { return _region; }

	inline void set_neighbors(std::vector<Superpixel*> input) { _neighbors = input; }
	inline void add_neighbor(Superpixel* input) { _neighbors.push_back(input); }
	inline void set_region(Region* input) { _region = input; }
	inline void set_colors(std::vector<cv::Vec3b> input) { _colors = input; }
	inline void set_points(std::vector<cv::Point> input) { _points = input; }
	inline void set_parent(Field* input) { _parent = input; }
	inline void add_point(cv::Point input) { _points.push_back(input); }
	inline void add_color(cv::Vec3b input) { _colors.push_back(input); }

	inline int num_points() { return _points.size(); }
#pragma endregion



private:
	Field* _parent;
	std::vector<cv::Point> _points;
	cv::Point _mean;
	std::vector<cv::Vec3b> _colors;
	cv::Vec3b _average_color; 
	cv::Mat _histogram;

	std::vector<Superpixel*> _neighbors;
	Region* _region;
};