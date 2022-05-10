#pragma once
#include"../../includes.h"





class Moment {
public:
	Moment();



#pragma region virtual
	virtual void set_past(Moment* input) = 0;
	virtual void set_future(Moment* input) = 0;
#pragma endregion

#pragma region get-set
	int moment_id() { return _moment_id; }
	Event* event_parent() { return _event_parent; }

	void set_moment_id(int input) { _moment_id = input; }
	void set_event_parent(Event* input) { _event_parent = input; }
#pragma endregion

private:
	int _moment_id;
	Event* _event_parent = nullptr;
};