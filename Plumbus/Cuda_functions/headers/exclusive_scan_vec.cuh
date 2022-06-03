#pragma once
#include"../../cuda_includes.h"














void exclusive_scan_vec_launch(gMat &input, gMat &output, int* sum );