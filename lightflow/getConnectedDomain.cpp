#include "getConnectedDomain.h"
#include <stack>

using namespace std;
using namespace cv;

void getConnectedDomain(Mat& src, vector<Rect>& boundingbox)//boundingboxΪ���ս������Ÿ�����ͨ��İ�Χ��
{
	int img_row = src.rows;
	int img_col = src.cols;
	Mat flag = Mat::zeros(Size(img_col, img_row), CV_8UC1);//��־����Ϊ0��ǰ���ص�δ���ʹ�
	for (int i = 0; i < img_row; i++)
	{
		for (int j = 0; j < img_col; j++)
		{
			if (src.ptr<uchar>(i)[j] == 0 && flag.ptr<uchar>(i)[j] == 0)
			{
				stack<Point2f> cd;
				cd.push(Point2f(j, i));
				flag.ptr<uchar>(i)[j] = 1;
				int minRow = i, minCol = j;//��Χ�����ϱ߽�
				int maxRow = i, maxCol = j;//��Χ���ҡ��±߽�
				while (!cd.empty())
				{
					Point2f tmp = cd.top();
					if (minRow > tmp.y)//���°�Χ��
						minRow = tmp.y;
					if (minCol > tmp.x)
						minCol = tmp.x;
					if (maxRow < tmp.y)
						maxRow = tmp.y;
					if (maxCol < tmp.x)
						maxCol = tmp.x;
					cd.pop();
					Point2f p[4];//�������ص㣬�����õ�������
					p[0] = Point2f(tmp.x - 1 > 0 ? tmp.x - 1 : 0, tmp.y);
					p[1] = Point2f(tmp.x + 1 < img_col - 1 ? tmp.x + 1 : img_row - 1, tmp.y);
					p[2] = Point2f(tmp.x, tmp.y - 1 > 0 ? tmp.y - 1 : 0);
					p[3] = Point2f(tmp.x, tmp.y + 1 < img_row - 1 ? tmp.y + 1 : img_row - 1);
					for (int m = 0; m < 4; m++)
					{
						int x = p[m].y;
						int y = p[m].x;
						if (src.ptr<uchar>(x)[y] == 0 && flag.ptr<uchar>(x)[y] == 0)//���δ���ʣ�����ջ������Ƿ��ʹ��õ�
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