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
#include <opencv2/cudaarithm.hpp>

//superpixels
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc.hpp>
#include<opencv2/tracking.hpp>

//thrust 
#include <thrust/pair.h>

//std
#include"standard_includes.h"

//cublas 
#include <cublas_v2.h>

namespace fs = std::filesystem;
