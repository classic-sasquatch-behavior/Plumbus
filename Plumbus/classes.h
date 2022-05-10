#pragma once








//events
#include"headers/events/Activity.h"
#include"headers/events/Event.h"
#include"headers/events/Shot.h"

//moments
#include"headers/moments/Antiframe.h"
#include"headers/moments/Frame.h"
#include"headers/moments/Moment.h"

//objects
#include"headers/objects/Object.h"
#include"headers/objects/Field.h"
#include"headers/objects/Region.h"
#include"headers/objects/Superpixel.h"

//structure
#include"headers/structure/Clip.h"
#include"headers/structure/Scene.h"
#include"headers/structure/VirtualEvent.h"

//util
#include "headers/util/ColorWheel.h"
#include "headers/util/Link.h"
#include"headers/util/Timer.h"
#include"headers//util/Window.h"

#include"Cuda_functions/headers/CudaInterface.h"

extern Timer* timer;
extern ColorWheel* colorwheel;
extern CudaInterface* GPU;



