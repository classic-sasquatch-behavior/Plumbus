#include"../../includes.h"


class ColorWheel {
public:
	ColorWheel();
	cv::Vec3b get_color(int STEP_SIZE);
private:
	int _color[3] = { 0,0,0 };
};






