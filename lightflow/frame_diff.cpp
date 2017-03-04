
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#define threshold_diff1 10 //设置简单帧差法阈值
#define threshold_diff2 10 //设置简单帧差法阈值
int counter = 0;
using namespace cv;
using namespace std;
int frame_diff(Mat img_src1, Mat img_src2, Mat img_src3,  Mat img)
{
	try
	{
		//Mat img_src1, img_src2, img_src3;//3帧法需要3帧图片
		Mat img_dst, gray1, gray2, gray3;
		Mat gray_diff1, gray_diff2;//存储2次相减的图片
		Mat gray;//用来显示前景的
		//Mat img;//显示凸包

		bool pause = false;

		//VideoCapture vido_file("1.1.avi");//在这里改相应的文件名
		//namedWindow("foreground", 0);


				//vido_file >> img_src1;
				//cvtColor(img_src1, gray1, CV_BGR2GRAY);
				//waitKey(5);

				//vido_file >> img_src2;
				//cvtColor(img_src2, gray2, CV_BGR2GRAY);

				//imshow("video_src", img_src2);//
				//waitKey(5);

				//vido_file >> img_src3;
				//cvtColor(img_src3, gray3, CV_BGR2GRAY);
				gray1 = img_src1;
				gray2 = img_src2;
				gray3 = img_src3;
				subtract(gray2, gray1, gray_diff1);//第二帧减第一帧
				subtract(gray3, gray2, gray_diff2);//第三帧减第二帧

				for (int i = 0; i<gray_diff1.rows; i++)
					for (int j = 0; j<gray_diff1.cols; j++)
					{
						if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
							gray_diff1.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
						else gray_diff1.at<unsigned char>(i, j) = 0;

						if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
							gray_diff2.at<unsigned char>(i, j) = 255;
						else gray_diff2.at<unsigned char>(i, j) = 0;
					}
				bitwise_and(gray_diff1, gray_diff2, gray);
				//imshow("foreground", gray);

				vector<Point> points;

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

				//计算凸包 
				vector<int> hull;
				convexHull(Mat(points), hull, true);
				int count = (int)hull.size();
				Point pt0 = points[hull[count - 1]];
				img = Mat::zeros(gray1.size(), CV_8UC1);
				//画凸包 
				Point *pt = new Point[count];
				for (int i = 0; i < count; i++)
				{
					pt[i] = points[hull[i]];
					//line(img, pt0, pt, Scalar(0, 255, 0), 4);
					//pt0 = pt;
				}
				const Point* ppt[1] = { pt };
				int npt[] = { count };
				fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255));
				delete[] pt;

				imshow("hull", img);
				int c = waitKey(100);

			}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		printf("%s\n", err_msg);
	}
	return 0;

}