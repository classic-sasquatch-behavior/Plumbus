#pragma once

//#include<cuda.h>
//#include<SDL.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include<opencv2/core/mat.hpp>

//superpixels
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc.hpp>
#include<opencv2/tracking.hpp>

//CUDA superpixel from github
#include"CUDAslic/include/SlicCudaHost.h"

//std
#include<string>
#include<filesystem>
#include<iostream>
#include<fstream>
#include<vector>
#include<memory>
#include<set>
#include<algorithm>
#include<queue>
#include<unordered_map>
#include<chrono>

namespace fs = std::filesystem;
