
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#define threshold_diff1 10 //���ü�֡���ֵ
#define threshold_diff2 10 //���ü�֡���ֵ
int counter = 0;
using namespace cv;
using namespace std;
int frame_diff(Mat img_src1, Mat img_src2, Mat img_src3,  Mat img)
{
	try
	{
		//Mat img_src1, img_src2, img_src3;//3֡����Ҫ3֡ͼƬ
		Mat img_dst, gray1, gray2, gray3;
		Mat gray_diff1, gray_diff2;//�洢2�������ͼƬ
		Mat gray;//������ʾǰ����
		//Mat img;//��ʾ͹��

		bool pause = false;

		//VideoCapture vido_file("1.1.avi");//���������Ӧ���ļ���
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
				subtract(gray2, gray1, gray_diff1);//�ڶ�֡����һ֡
				subtract(gray3, gray2, gray_diff2);//����֡���ڶ�֡

				for (int i = 0; i<gray_diff1.rows; i++)
					for (int j = 0; j<gray_diff1.cols; j++)
					{
						if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
							gray_diff1.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
						else gray_diff1.at<unsigned char>(i, j) = 0;

						if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
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

				//����͹�� 
				vector<int> hull;
				convexHull(Mat(points), hull, true);
				int count = (int)hull.size();
				Point pt0 = points[hull[count - 1]];
				img = Mat::zeros(gray1.size(), CV_8UC1);
				//��͹�� 
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