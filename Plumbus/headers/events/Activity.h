#pragma once
#include"../../includes.h"
#include"Event.h"
#include"Shot.h"


class Clip;


class Activity : public Event {
public:
	Activity(int beginning, int end, Clip* parent);

private:
};