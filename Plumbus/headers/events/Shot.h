#pragma once
#include"../../includes.h"
#include"Event.h"
#include"Activity.h"

class Clip;

class Shot : public Event{
public:
	Shot(int beginning, int end, Clip* parent);
	~Shot();
	void find_objects_within_frames();
	void link_objects_between_frames();

#pragma region get-set
	int shot_id() { return _shot_id; }
	bool hard_cut_is_before() { return _hard_cut_before; }
	bool hard_cut_is_after() { return _hard_cut_after; }

	void set_hard_cut_before(bool input) { _hard_cut_before = input; }
	void set_hard_cut_after(bool input) { _hard_cut_after = input; }
	void set_shot_id(int input) { _shot_id = input; }
#pragma endregion


private:
	int _shot_id;
	bool _hard_cut_before = false;
	bool _hard_cut_after = false;
};