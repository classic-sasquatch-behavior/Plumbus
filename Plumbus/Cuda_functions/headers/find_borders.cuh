#pragma once
#include"../../cuda_includes.h"

std::vector<thrust::pair<int, int>> find_borders_launch(int src_width, int src_height, cv::Mat d_src);