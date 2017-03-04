#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stack>

using namespace std;
using namespace cv;
void getConnectedDomain(Mat& src, vector<Rect>& boundingbox);//boundingbox为最终结果，存放各个连通域的包围盒

