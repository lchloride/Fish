#include <cmath>
#include <iostream>
#include "areaStatistic.h"

AreaStatistic::AreaStatistic() {
	avg = 0;
	var = 0;
	count = 0;
}

void AreaStatistic::pushArea(double area) {
	count++;
	double new_avg = (avg*(count - 1) + area) / count;
	double new_var;
	if (count >= 2)
		new_var = (count - 2)*var / (count - 1) + pow((area - avg), 2) / count;
	else
		new_var = 0;
	avg = new_avg;
	var = new_var;
}

double AreaStatistic::getAvg() {
	return avg;
}

double AreaStatistic::getVar() {
	return var;
}

int AreaStatistic::getCount() {
	return count;
}

bool AreaStatistic::checkRange(double area) {
	if (area >= avg - 3 * sqrt(var) && area <= avg + 3 * sqrt(var))
		return true;
	else
		return false;
}
