
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#define threshold_diff1 10 //设置简单帧差法阈值
#define threshold_diff2 10 //设置简单帧差法阈值
using namespace cv;
using namespace std;
int main(int argc, unsigned char* argv[])
{
	try
	{
		Mat img_src1, img_src2, img_src3;//3帧法需要3帧图片
		Mat img_dst, gray1, gray2, gray3;
		Mat gray_diff1, gray_diff2;//存储2次相减的图片
		Mat gray;//用来显示前景的
		Mat img;//显示凸包

		vector<Point> points;
		bool pause = false;

		VideoCapture vido_file("1.1.avi");//在这里改相应的文件名
		namedWindow("foreground", 0);
		for (;;)
		{
			if (!false)
			{
				/*读取视频中连续三帧*/
				vido_file >> img_src1;
				cvtColor(img_src1, gray1, CV_BGR2GRAY);
				waitKey(5);

				vido_file >> img_src2;
				cvtColor(img_src2, gray2, CV_BGR2GRAY);

				imshow("video_src", img_src2);//
				waitKey(5);

				vido_file >> img_src3;
				cvtColor(img_src3, gray3, CV_BGR2GRAY);

				/*帧差法开始*/
				
				subtract(gray2, gray1, gray_diff1);//第二帧减第一帧
				subtract(gray3, gray2, gray_diff2);//第三帧减第二帧
				
				/*
				  gray_diff1和gray_diff2分别为一二和三四的帧差
				  下面要根据与阈值的大小关系进行二值化处理	
				  大于阈值即表示为可见的变化，在我们处理后的图形中用白色显示；
				  小于阈值认为可以认为没有变化，用黑色显示
				*/

				points.clear();

				for (int i = 0; i<gray_diff1.rows; i++)
					for (int j = 0; j<gray_diff1.cols; j++)
					{
						if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//
							gray_diff1.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
						else gray_diff1.at<unsigned char>(i, j) = 0;

						if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
							gray_diff2.at<unsigned char>(i, j) = 255;
						else gray_diff2.at<unsigned char>(i, j) = 0;
					}
				/*我们所用的图像表示了连续两帧中都发生变化的部分（与）*/
				bitwise_and(gray_diff1, gray_diff2, gray);

				/*显示帧差法结果*/
				imshow("foreground", gray);

				/*为便于后续计算，求出帧差法所找到的部分的凸包*/
				
				/*求出白色部分的所有点*/
				for (int i = 0; i < gray.rows; i++)
				{
					for (int j = 0; j < gray.cols; j++)
					{
						Point pt;
						if (gray.at<unsigned char>(i, j) == 255)
						{
							pt.x = j;
							pt.y = i;
							points.push_back(pt);
						}
					}			
				}

				img = Mat::zeros(gray.size(),CV_8UC3);

				//计算凸包 
				vector<int> hull;
				convexHull(Mat(points), hull, true);
				int count = (int)hull.size();//凸包中点的个数
				if (count == 0)
					continue;
				Point pt0 = points[hull[count - 1]];
				img = img_src1.clone();
				//画凸包
				for (int i = 0; i < count; i++)
				{
					Point pt = points[hull[i]];
					line(img, pt0, pt, Scalar(0, 0,255), 4);
					pt0 = pt;
				}
				/*显示凸包*/
				imshow("hull", img);
				

			}
			char c = (char)waitKey(10);
			if (c == 27)
			{
				break;
			}
		}
		
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		printf("%s\n", err_msg);
	}
	return 0;

}