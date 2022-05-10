#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"





Region::Region(Field* parent) {
	_parent = parent;
}

Region::~Region() {
}

cv::Point Region::id(){ return _constituents[0]->mean(); }
