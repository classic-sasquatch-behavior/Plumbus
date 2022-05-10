#pragma once
#include"../../includes.h"

class Field;
class Superpixel;


class Region {
public:
	Region(Field* parent);
	~Region();
	inline void clear_constituents() { _constituents.clear(); }

#pragma region get-set

	inline std::vector<Superpixel*> all_constituents() { return _constituents; }
	inline Field* parent() { return _parent; }
	inline cv::Vec3b average_color() { return _average_color; }
	cv::Point id();

	inline void set_average_color(cv::Vec3b input) { _average_color = input; }
	inline void add_constituent(Superpixel* input) { _constituents.push_back(input); }
	inline void set_parent(Field* input) { _parent = input; }

	inline int num_constituents() { return _constituents.size(); }
#pragma endregion





private:
	std::vector<Superpixel*> _constituents;
	Field* _parent;
	cv::Vec3b _average_color;
};