#include "getConnectedDomain.h"
#include <stack>

using namespace std;
using namespace cv;

void getConnectedDomain(Mat& src, vector<Rect>& boundingbox)//boundingbox为最终结果，存放各个连通域的包围盒
{
	int img_row = src.rows;
	int img_col = src.cols;
	Mat flag = Mat::zeros(Size(img_col, img_row), CV_8UC1);//标志矩阵，为0则当前像素点未访问过
	for (int i = 0; i < img_row; i++)
	{
		for (int j = 0; j < img_col; j++)
		{
			if (src.ptr<uchar>(i)[j] == 0 && flag.ptr<uchar>(i)[j] == 0)
			{
				stack<Point2f> cd;
				cd.push(Point2f(j, i));
				flag.ptr<uchar>(i)[j] = 1;
				int minRow = i, minCol = j;//包围盒左、上边界
				int maxRow = i, maxCol = j;//包围盒右、下边界
				while (!cd.empty())
				{
					Point2f tmp = cd.top();
					if (minRow > tmp.y)//更新包围盒
						minRow = tmp.y;
					if (minCol > tmp.x)
						minCol = tmp.x;
					if (maxRow < tmp.y)
						maxRow = tmp.y;
					if (maxCol < tmp.x)
						maxCol = tmp.x;
					cd.pop();
					Point2f p[4];//邻域像素点，这里用的四邻域
					p[0] = Point2f(tmp.x - 1 > 0 ? tmp.x - 1 : 0, tmp.y);
					p[1] = Point2f(tmp.x + 1 < img_col - 1 ? tmp.x + 1 : img_row - 1, tmp.y);
					p[2] = Point2f(tmp.x, tmp.y - 1 > 0 ? tmp.y - 1 : 0);
					p[3] = Point2f(tmp.x, tmp.y + 1 < img_row - 1 ? tmp.y + 1 : img_row - 1);
					for (int m = 0; m < 4; m++)
					{
						int x = p[m].y;
						int y = p[m].x;
						if (src.ptr<uchar>(x)[y] == 0 && flag.ptr<uchar>(x)[y] == 0)//如果未访问，则入栈，并标记访问过该点
						{
							cd.push(p[m]);
							flag.ptr<uchar>(x)[y] = 1;
						}
					}
				}
				Rect rect(Point2f(minCol, minRow), Point2f(maxCol + 1, maxRow + 1));
				boundingbox.push_back(rect);
			}
		}
	}
}