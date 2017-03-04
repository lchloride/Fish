#include "selectionsort.h"



int getMaxIdx(vector<Contours> src, int st, int ed)
{
	int max_idx = st;
	for (int i=st; i<ed; i++)
		if (src[i].area > src[max_idx].area)
			max_idx = i;
	return max_idx;
}

void swap(std::vector<Contours>& src, int x, int y)
{
	Contours temp;
	temp.setContours(src[x].area, src[x].index);
	src[x].setContours(src[y].area, src[y].index);
	src[y].setContours(temp.area, temp.index);
}

int selectionSort(vector<Contours>& src, int range)
{
	int st = 0;
	int ed = src.size();
	if (range > src.size())
		range = src.size();
	// Select top range values
	for (int i=0; i<range; i++) {
		int max_idx = getMaxIdx(src, st, ed);
		swap(src, max_idx, i);
		st++;
	}
	return range;
}

//int main(int argc, char const *argv[])
//{
//	std::vector<Contours> contours;
//	Contours c;
//	c.setContours(1, 0);
//	contours.push_back(c);
//	c.setContours(3, 1);
//	contours.push_back(c);
//	c.setContours(1, 2);
//	contours.push_back(c);
//	c.setContours(2, 3);
//	contours.push_back(c);
//	selectionSort(contours, 2);
//	for (int i=0; i<contours.size(); i++)
//		cout<<contours[i].area<<","<<contours[i].index<<endl;
//	return 0;
//}