#pragma once
#include <iostream>
#include <vector>
using namespace std;
class Contours {
public:
	double area;
	int index;
	void setContours(double a, int idx) { area = a; index = idx; }
	void display() {
		cout << "(" << area << ", " << index << ") ";
	}
};
int selectionSort(std::vector<Contours>& src, int range);