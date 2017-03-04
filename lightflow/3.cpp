
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#define threshold_diff1 10 //���ü�֡���ֵ
#define threshold_diff2 10 //���ü�֡���ֵ
using namespace cv;
using namespace std;
int main(int argc, unsigned char* argv[])
{
	try
	{
		Mat img_src1, img_src2, img_src3;//3֡����Ҫ3֡ͼƬ
		Mat img_dst, gray1, gray2, gray3;
		Mat gray_diff1, gray_diff2;//�洢2�������ͼƬ
		Mat gray;//������ʾǰ����
		Mat img;//��ʾ͹��

		vector<Point> points;
		bool pause = false;

		VideoCapture vido_file("1.1.avi");//���������Ӧ���ļ���
		namedWindow("foreground", 0);
		for (;;)
		{
			if (!false)
			{
				/*��ȡ��Ƶ��������֡*/
				vido_file >> img_src1;
				cvtColor(img_src1, gray1, CV_BGR2GRAY);
				waitKey(5);

				vido_file >> img_src2;
				cvtColor(img_src2, gray2, CV_BGR2GRAY);

				imshow("video_src", img_src2);//
				waitKey(5);

				vido_file >> img_src3;
				cvtColor(img_src3, gray3, CV_BGR2GRAY);

				/*֡���ʼ*/
				
				subtract(gray2, gray1, gray_diff1);//�ڶ�֡����һ֡
				subtract(gray3, gray2, gray_diff2);//����֡���ڶ�֡
				
				/*
				  gray_diff1��gray_diff2�ֱ�Ϊһ�������ĵ�֡��
				  ����Ҫ��������ֵ�Ĵ�С��ϵ���ж�ֵ������	
				  ������ֵ����ʾΪ�ɼ��ı仯�������Ǵ�����ͼ�����ð�ɫ��ʾ��
				  С����ֵ��Ϊ������Ϊû�б仯���ú�ɫ��ʾ
				*/

				points.clear();

				for (int i = 0; i<gray_diff1.rows; i++)
					for (int j = 0; j<gray_diff1.cols; j++)
					{
						if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//
							gray_diff1.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
						else gray_diff1.at<unsigned char>(i, j) = 0;

						if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
							gray_diff2.at<unsigned char>(i, j) = 255;
						else gray_diff2.at<unsigned char>(i, j) = 0;
					}
				/*�������õ�ͼ���ʾ��������֡�ж������仯�Ĳ��֣��룩*/
				bitwise_and(gray_diff1, gray_diff2, gray);

				/*��ʾ֡����*/
				imshow("foreground", gray);

				/*Ϊ���ں������㣬���֡����ҵ��Ĳ��ֵ�͹��*/
				
				/*�����ɫ���ֵ����е�*/
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

				//����͹�� 
				vector<int> hull;
				convexHull(Mat(points), hull, true);
				int count = (int)hull.size();//͹���е�ĸ���
				if (count == 0)
					continue;
				Point pt0 = points[hull[count - 1]];
				img = img_src1.clone();
				//��͹��
				for (int i = 0; i < count; i++)
				{
					Point pt = points[hull[i]];
					line(img, pt0, pt, Scalar(0, 0,255), 4);
					pt0 = pt;
				}
				/*��ʾ͹��*/
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