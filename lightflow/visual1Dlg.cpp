
// visual1Dlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "visual1.h"
#include "visual1Dlg.h"
#include "afxdialogex.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvvImage.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;

using namespace std;

#define TM_MSG 10000000
#define WINDOW_NAME "ʶ��Ч��ͼ"        //Ϊ���ڱ��ⶨ��ĺ� 
#define AREA_LIMIT 40 //Ϊ��ʶ��ͷ�������е���ֵ���У������ʶ��������������С

Mat g_srcImage, g_grayImage, g_binImage;//�洢ԭͼ�Լ���Ӧ�ĻҶ�ͼ�Ͷ�ֵ��ͼ
int g_maxCornerNumber = 2; //��ȡ�ǵ�����ֵ��Ĭ��Ϊ2
RNG g_rng(12345);//��ʼ�������������
int framenum = 0; //�洢�Ѷ�ȡ��֡������
const int thres = 13; //Ϊ��ʶ��ͷ�������е���ֵ����������ֵ(Ӧ����Ҫ��ѧϰ)
const int big_thres = 33; //Ϊ��ʶ��β�������е���ֵ����������ֵ(Ӧ����Ҫ��ѧϰ)
vector<vector<Point> > contours; // Ϊ��ʶ��ͷ�������е���ֵ������
vector<vector<Point> > big_contours; //Ϊ��ʶ��β�������е���ֵ������
int contour_num = 1; //contours��ʵ��������������Ķ�Ӧ�±�
Mat mask; //��Ĥ��������ȡROI
int ds = 20; //ʶ���ƫ�ƣ���֤��������ܹ���ʶ�����
int median_thres = 13;



//ʹ�ýǵ����㷨������������нǵ��⣬�õ���������Ϊͷβ��
//����ȡͷβ�����꣬������ԭ���ԭ��
void GoodFeaturesToTrack()
{
	g_maxCornerNumber = 2;

	//Shi-Tomasi�㷨��goodFeaturesToTrack�������Ĳ���׼��
	vector<Point2f> corners;
	double qualityLevel = 0.01;//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 50;//�ǵ�֮�����С����(�Ժ���Ҫ��ѧϰ)
	int blockSize = 7;//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;//Ȩ��ϵ��
	Mat copy = g_srcImage.clone();	//����Դͼ��һ����ʱ�����У�Ϊ�����

									//��ֵ��
	threshold(g_grayImage, g_binImage, big_thres, 255, THRESH_BINARY);

	//Ѱ������
	vector<Vec4i> hierarchy;
	findContours(g_binImage, big_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (big_contours.size() == 0) //û�ҵ������򲻽��к�������
		return;

	Mat draw = Mat::zeros(g_binImage.size(), CV_8UC1);
	//cout << "Area: ";
	double maxArea = 0; //��ֵ�����ͼ�ε�������
	int big_contour_num = 0; //��������Ӧ�������±�
	for (int i = 1; i < big_contours.size(); i++) //����ֵ����������������
	{
		//drawContours(draw, big_contours, i, Scalar(255, 0, 0));
		double area = contourArea(big_contours[i]);
		if (area > maxArea)
		{
			big_contour_num = i;
			maxArea = area;
		}
	}
	//���µ�һ��Ϊ�����ã��������ս��
	if (big_contour_num > 0)//���������������ͼ
		drawContours(draw, big_contours, big_contour_num, Scalar(0, 255, 0));

	//���������������ľ��Σ�Ȼ���øþ��β�����Ĥmask
	RotatedRect box = minAreaRect(big_contours[big_contour_num]);//��������������ת����
	Point2f vertex[4];
	Rect rect = box.boundingRect();//����ת���ε���Ӿ��Σ���Ϊ���ο�
								   //���ݾ��ο�ȷ�������ĸ����������
	Point rect_vertex[] = { Point(rect.x - ds < 0 ? 0 : rect.x - ds, rect.y - ds < 0 ? 0 : rect.y - ds),
		Point((rect.x + rect.width + ds >= g_grayImage.cols) ? g_grayImage.cols - 1 : rect.x + rect.width + ds,
			rect.y - ds < 0 ? 0 : rect.y - ds),
		Point((rect.x + rect.width + ds >= g_grayImage.cols) ? g_grayImage.cols - 1 : rect.x + rect.width + ds,
			(rect.y + rect.height + ds >= g_grayImage.rows) ? g_grayImage.rows - 1 : rect.y + rect.height + ds),
		Point(rect.x - ds < 0 ? 0 : rect.x - ds,
			(rect.y + rect.height + ds >= g_grayImage.rows) ? g_grayImage.rows - 1 : rect.y + rect.height + ds)
	};

	for (int i = 0; i < 4; i++) //�����ο������ͼƬ��
	{
		//rect_vertex[i].x = (rect_vertex[i].x < 0) ? 0 : rect_vertex[i].x;
		//rect_vertex[i].x = (rect_vertex[i].x > g_grayImage.cols) ? g_grayImage.cols - 1 : rect_vertex[i].x;

		//rect_vertex[i].y = (rect_vertex[i].y < 0) ? 0 : rect_vertex[i].y;
		//rect_vertex[i].y = (rect_vertex[i].y > g_grayImage.rows) ? g_grayImage.rows - 1 : rect_vertex[i].y;

		line(copy, rect_vertex[i], rect_vertex[(i + 1) % 4], Scalar(0, 0, 0), 2, LINE_AA);
	}
	mask = Mat::zeros(g_grayImage.size(), CV_8UC1); //��Ĥ��ʼ��
	if (rect.height > 0 && rect.width>0)
	{
		if (rect.x + rect.width >= g_grayImage.cols)
			rect.width = g_grayImage.cols - rect.x - 1;
		if (rect.y + rect.height >= g_grayImage.rows)
			rect.height = g_grayImage.rows - rect.y - 1;
		mask(rect).setTo(255); //����Ĥ��ROI��Ӧ�Ĳ�����Ϊ����Ȥ����
	}
	else
		mask.setTo(255);

	//����Shi-Tomasi�ǵ���
	goodFeaturesToTrack(g_grayImage,//����ͼ��
		corners,//��⵽�Ľǵ���������
		g_maxCornerNumber,//�ǵ���������
		qualityLevel,//�ǵ���ɽ��ܵ���С����ֵ
		minDistance,//�ǵ�֮�����С����
		mask,//����Ȥ����
		blockSize,//���㵼������ؾ���ʱָ��������Χ
		false,//��ʹ��Harris�ǵ���
		k);//Ȩ��ϵ��


		   //���������Ϣ
		   //cout << "\t>�˴μ�⵽�Ľǵ�����Ϊ��" << corners.size() << endl;

		   //
	int r = 4;
//	FILE *fp;
//	fp = fopen("result.txt", "a");
	Point2f midpoint;
	vector<double> point2contour; //��Žǵ㵽���������ľ���
	point2contour.assign(corners.size(), 0);
	int min_num = -1; //��������������Ľǵ���±�
	for (int i = 0; i < corners.size(); i++)
	{
		//���������ɫ���Ƴ��ǵ�
		circle(copy, corners[i], r, Scalar(g_rng.uniform(0, 200), g_rng.uniform(0, 200),
			g_rng.uniform(0, 200)), -1, 8, 0);
		//fprintf(fp, "(%0.0f, %0.0f)  ", corners[i].x, corners[i].y);
		double t = pointPolygonTest(contours[contour_num], Point2f(corners[i].x, corners[i].y), true); //��㵽���������ľ���
																									   //if (t >= 0) //���������ڣ�������0
																									   //	point2contour[i] = 0;
																									   //else //���������ڣ�������|t|
																									   //	point2contour[i] = -t;
		point2contour[i] = -t;

		if (min_num == -1) //��������Сֵ
			min_num = 0;
		else
			if (point2contour[i] < point2contour[min_num])
				min_num = i;
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);
	if (min_num > -1)
	{
//		fprintf(fp, "(%0.0lf, %0.0lf), ", corners[min_num].x, corners[min_num].y); //��ͷ������д���ļ�
		circle(copy, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);	 //�԰�ɫ���Ƴ�ͷ�ĵ�
	//	if (corners.size() > 1)
			//fprintf(fp, "(%0.0lf, %0.0lf)\n", corners[!min_num].x, corners[!min_num].y); //��β������д���ļ�
	//	else
		//	fputc('\n', fp);
	}
	//fclose(fp);

	//��ʾ�����£�ʶ��������
	imshow(WINDOW_NAME, copy);
}





// Cvisual1Dlg �Ի���







Cvisual1Dlg::Cvisual1Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_VISUAL1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Cvisual1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(Cvisual1Dlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &Cvisual1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &Cvisual1Dlg::OnBnClickedButton2)
END_MESSAGE_MAP()


// Cvisual1Dlg ��Ϣ�������

BOOL Cvisual1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void Cvisual1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR Cvisual1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void Cvisual1Dlg::OnBnClickedButton1()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	//IplImage *src; // ����IplImageָ�����src     
	//  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
	//  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
	//  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
	//  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
	//  cvDestroyWindow("me");//���ٴ���lena    
	//  cvReleaseImage(&src);//�ͷ�IplImageָ��src   


	CDC *pDC = GetDlgItem(IDC_ORIGIN)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	CRect rect;
	// ������
	GetDlgItem(IDC_ORIGIN)->GetClientRect(&rect); //��ȡbox1�ͻ���
	
	CvCapture *capture = cvCreateFileCapture("top_sample.avi");  //��ȡ��Ƶ
	if (capture == NULL) {
		printf("NO capture");    //��ȡ���ɹ������ʶ
								 //return 1;
	};
	double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);   //��ȡ��Ƶ��֡��
	int vfps = 1000 / fps;                                        //����ÿ֡���ŵ�ʱ��
	//printf("%5.1f\t%5d\n", fps, vfps);
	double frames = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ���ж���֡
	//printf("frames is %f\n", frames);
	//cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //���崰��
	IplImage *frame;
	CvvImage cimg;

//	int i = 0;
	while (1) {
		//for (int i = 0;i < 1000000;++i);
		frame = cvQueryFrame(capture);  
		if (!frame)break;                        //ץȡ֡
		cimg.CopyOf(frame, frame->nChannels);
		cimg.DrawToHDC(hdc, &rect);
		//KillTimer(TM_MSG);
		//SetTimer(TM_MSG, 100000000, NULL);
	//	float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
	//	printf("%f\n", ratio);
		
		//cvShowImage("IDC_STATIC",frame);                          //��ʾ
		//++i;
		char c = cvWaitKey(vfps);
		if (c == 27)break;

		
	}
	ReleaseDC(pDC);
	cvReleaseCapture(&capture);
	//cvDestroyWindow("example");

}


void Cvisual1Dlg::OnBnClickedButton2()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	VideoCapture capture("top_sample.avi");
	if (capture.isOpened())	// ����ͷ��ȡ�ļ�����
	{
		while (true) //ѭ����ȡ��Ƶ
		{
			//����Դͼ�񲢽���ת��Ϊ�Ҷ�ͼ
			capture >> g_srcImage;

			if (g_srcImage.empty()) //�޷���ȡ��Ƶ
			{
				printf(" No captured frame -- Break!");
				break;
			}

			framenum++;

			//���Ը�ʴ����
			Mat ele = getStructuringElement(MORPH_RECT, Size(5, 3));
			erode(g_srcImage, g_srcImage, ele);

			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);//ת�ɻҶ�ͼ

															  //��ֵ�˲���ȥ���޹ص�����
			medianBlur(g_grayImage, g_grayImage, median_thres);

			//��ֵ�����õ������ɵ�����ͼ��
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);

			//��ȡ����ͼ�ε�����
			vector<Vec4i> hierarchy;
			findContours(g_binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//Mat draw = Mat::zeros(g_binImage.size(), CV_8UC3);
			//Mat draw = g_binImage.clone();
			//cvtColor(draw, draw, COLOR_GRAY2BGR);
			//cout << "Area: ";
			double maxArea = 0;
			contour_num = 0;
			for (int i = 1; i < contours.size(); i++)//��ȡ�������У�����ͼ����Ե������������������
			{
				//drawContours(draw, contours, i, Scalar(255, 0, 0));
				double area = contourArea(contours[i]);
				if (area > maxArea)
				{
					contour_num = i;
					maxArea = area;
				}
				//cout << area << " ";
			}
			//cout << endl;
			//imshow("draw", draw);
			//int cc = waitKey(10);
			if (contours.size() > 2)
			{
				//cout << "contour num = " << contour_num << endl;
				//system("pause");
			}


			//�������ڣ���ͼ����нǵ�ʶ��
			if (maxArea >= AREA_LIMIT)//���õ�����������Ƚ�С��ʱ�򣬲��ܱ�֤�㷨�ĳ�������������
			{
				namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
				imshow(WINDOW_NAME, g_srcImage);
				GoodFeaturesToTrack();
			}
			int c = waitKey(10);
			if ((char)c == 27)
			{
				break;
			}
		}
	}
}
