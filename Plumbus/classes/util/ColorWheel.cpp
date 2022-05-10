#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"








ColorWheel::ColorWheel() {

}

cv::Vec3b ColorWheel::get_color(int STEP_SIZE) {





	if (_color[2] < 256) {
		_color[2] += STEP_SIZE;
	} else {
		_color[2] = 0;
		if (_color[1] < 256) {
			_color[1] += STEP_SIZE;
		} else {
			_color[1] = 0;
			if (_color[0] < 256) {
				_color[0] += STEP_SIZE;
			}
			else {
				_color[0] = 0;
			}
		}
	}
	cv::Vec3b output = {uchar(std::max(_color[0] - 1, 0)), uchar(std::max(_color[1] - 1, 0)), uchar(std::max(_color[2] - 1,0))};
	return output;
}