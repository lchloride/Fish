#pragma once
#include <cmath>

class AreaStatistic {
private:
	double avg;
	double var;
	int count;
public:
	AreaStatistic();
	void pushArea(double area);
	double getAvg();
	double getVar();
	int getCount();
	bool checkRange(double area);
};