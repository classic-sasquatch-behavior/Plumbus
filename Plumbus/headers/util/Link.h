#pragma once
#include"../../includes.h"




class Link {
public:
	Link(Event* owner);

#pragma region get-set

	Event* owner() { return _owner; }
	Link* previous_link() { return _previous_link; }
	Link* next_link() { return _next_link; }

	void set_owner(Event* input) { _owner = input; }
	void set_previous_link(Link* input) { _previous_link = input; }
	void set_next_link(Link* input) { _next_link = input; }

#pragma endregion


private:
	Event* _owner = nullptr;
	Link* _previous_link = nullptr;
	Link* _next_link = nullptr;

};







