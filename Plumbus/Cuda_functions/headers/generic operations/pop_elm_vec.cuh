#pragma once
#include"../../../cuda_includes.h"












void pop_elm_vec_launch(gMat &input, gMat &output, int pop_elm, int* max = 0);