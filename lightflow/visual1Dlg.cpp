
// visual1Dlg.cpp : 实现文件
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
#define WINDOW_NAME "识别效果图"        //为窗口标题定义的宏 
#define AREA_LIMIT 40 //为了识别头部而进行的阈值化中，定义的识别区域最低面积大小

Mat g_srcImage, g_grayImage, g_binImage;//存储原图以及对应的灰度图和二值化图
int g_maxCornerNumber = 2; //获取角点的最大值，默认为2
RNG g_rng(12345);//初始化随机数生成器
int framenum = 0; //存储已读取的帧的数量
const int thres = 13; //为了识别头部而进行的阈值化操作的阈值(应该需要自学习)
const int big_thres = 33; //为了识别尾部而进行的阈值化操作的阈值(应该需要自学习)
vector<vector<Point> > contours; // 为了识别头部而进行的阈值化操作
vector<vector<Point> > big_contours; //为了识别尾部而进行的阈值化操作
int contour_num = 1; //contours中实际描述鱼的轮廓的对应下标
Mat mask; //掩膜，用于提取ROI
int ds = 20; //识别框偏移，保证鱼的轮廓能够在识别框中
int median_thres = 13;



//使用角点检测算法对鱼的轮廓进行角点检测，得到的特征点为头尾，
//并提取头尾的坐标，以坐标原点的原点
void GoodFeaturesToTrack()
{
	g_maxCornerNumber = 2;

	//Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	vector<Point2f> corners;
	double qualityLevel = 0.01;//角点检测可接受的最小特征值
	double minDistance = 50;//角点之间的最小距离(以后需要自学习)
	int blockSize = 7;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数
	Mat copy = g_srcImage.clone();	//复制源图像到一个临时变量中，为了输出

									//阈值化
	threshold(g_grayImage, g_binImage, big_thres, 255, THRESH_BINARY);

	//寻找轮廓
	vector<Vec4i> hierarchy;
	findContours(g_binImage, big_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (big_contours.size() == 0) //没找到轮廓则不进行后续操作
		return;

	Mat draw = Mat::zeros(g_binImage.size(), CV_8UC1);
	//cout << "Area: ";
	double maxArea = 0; //阈值化后的图形的最大面积
	int big_contour_num = 0; //最大面积对应的轮廓下标
	for (int i = 1; i < big_contours.size(); i++) //求阈值化后轮廓的最大面积
	{
		//drawContours(draw, big_contours, i, Scalar(255, 0, 0));
		double area = contourArea(big_contours[i]);
		if (area > maxArea)
		{
			big_contour_num = i;
			maxArea = area;
		}
	}
	//以下的一句为调试用，并非最终结果
	if (big_contour_num > 0)//已求出最大面积，绘图
		drawContours(draw, big_contours, big_contour_num, Scalar(0, 255, 0));

	//求出能拟合鱼轮廓的矩形，然后用该矩形残生掩膜mask
	RotatedRect box = minAreaRect(big_contours[big_contour_num]);//求出拟合轮廓的旋转矩形
	Point2f vertex[4];
	Rect rect = box.boundingRect();//求旋转矩形的外接矩形，即为矩形框
								   //根据矩形框，确定矩形四个顶点的坐标
	Point rect_vertex[] = { Point(rect.x - ds < 0 ? 0 : rect.x - ds, rect.y - ds < 0 ? 0 : rect.y - ds),
		Point((rect.x + rect.width + ds >= g_grayImage.cols) ? g_grayImage.cols - 1 : rect.x + rect.width + ds,
			rect.y - ds < 0 ? 0 : rect.y - ds),
		Point((rect.x + rect.width + ds >= g_grayImage.cols) ? g_grayImage.cols - 1 : rect.x + rect.width + ds,
			(rect.y + rect.height + ds >= g_grayImage.rows) ? g_grayImage.rows - 1 : rect.y + rect.height + ds),
		Point(rect.x - ds < 0 ? 0 : rect.x - ds,
			(rect.y + rect.height + ds >= g_grayImage.rows) ? g_grayImage.rows - 1 : rect.y + rect.height + ds)
	};

	for (int i = 0; i < 4; i++) //将矩形框绘制在图片中
	{
		//rect_vertex[i].x = (rect_vertex[i].x < 0) ? 0 : rect_vertex[i].x;
		//rect_vertex[i].x = (rect_vertex[i].x > g_grayImage.cols) ? g_grayImage.cols - 1 : rect_vertex[i].x;

		//rect_vertex[i].y = (rect_vertex[i].y < 0) ? 0 : rect_vertex[i].y;
		//rect_vertex[i].y = (rect_vertex[i].y > g_grayImage.rows) ? g_grayImage.rows - 1 : rect_vertex[i].y;

		line(copy, rect_vertex[i], rect_vertex[(i + 1) % 4], Scalar(0, 0, 0), 2, LINE_AA);
	}
	mask = Mat::zeros(g_grayImage.size(), CV_8UC1); //掩膜初始化
	if (rect.height > 0 && rect.width>0)
	{
		if (rect.x + rect.width >= g_grayImage.cols)
			rect.width = g_grayImage.cols - rect.x - 1;
		if (rect.y + rect.height >= g_grayImage.rows)
			rect.height = g_grayImage.rows - rect.y - 1;
		mask(rect).setTo(255); //将掩膜中ROI对应的部分设为感兴趣区域
	}
	else
		mask.setTo(255);

	//进行Shi-Tomasi角点检测
	goodFeaturesToTrack(g_grayImage,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		mask,//感兴趣区域
		blockSize,//计算导数自相关矩阵时指定的邻域范围
		false,//不使用Harris角点检测
		k);//权重系数


		   //输出文字信息
		   //cout << "\t>此次检测到的角点数量为：" << corners.size() << endl;

		   //
	int r = 4;
//	FILE *fp;
//	fp = fopen("result.txt", "a");
	Point2f midpoint;
	vector<double> point2contour; //存放角点到躯干轮廓的距离
	point2contour.assign(corners.size(), 0);
	int min_num = -1; //到躯干轮廓最近的角点的下标
	for (int i = 0; i < corners.size(); i++)
	{
		//以随机的颜色绘制出角点
		circle(copy, corners[i], r, Scalar(g_rng.uniform(0, 200), g_rng.uniform(0, 200),
			g_rng.uniform(0, 200)), -1, 8, 0);
		//fprintf(fp, "(%0.0f, %0.0f)  ", corners[i].x, corners[i].y);
		double t = pointPolygonTest(contours[contour_num], Point2f(corners[i].x, corners[i].y), true); //求点到躯干轮廓的距离
																									   //if (t >= 0) //点在轮廓内，距离是0
																									   //	point2contour[i] = 0;
																									   //else //点在轮廓内，距离是|t|
																									   //	point2contour[i] = -t;
		point2contour[i] = -t;

		if (min_num == -1) //求距离的最小值
			min_num = 0;
		else
			if (point2contour[i] < point2contour[min_num])
				min_num = i;
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);
	if (min_num > -1)
	{
//		fprintf(fp, "(%0.0lf, %0.0lf), ", corners[min_num].x, corners[min_num].y); //将头的坐标写入文件
		circle(copy, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);	 //以白色绘制出头的点
	//	if (corners.size() > 1)
			//fprintf(fp, "(%0.0lf, %0.0lf)\n", corners[!min_num].x, corners[!min_num].y); //将尾的坐标写入文件
	//	else
		//	fputc('\n', fp);
	}
	//fclose(fp);

	//显示（更新）识别结果窗口
	imshow(WINDOW_NAME, copy);
}





// Cvisual1Dlg 对话框







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


// Cvisual1Dlg 消息处理程序

BOOL Cvisual1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void Cvisual1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR Cvisual1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void Cvisual1Dlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
	//IplImage *src; // 定义IplImage指针变量src     
	//  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
	//  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
	//  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
	//  cvWaitKey(0);//无限等待，即图像总显示    
	//  cvDestroyWindow("me");//销毁窗口lena    
	//  cvReleaseImage(&src);//释放IplImage指针src   


	CDC *pDC = GetDlgItem(IDC_ORIGIN)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	HDC hdc = pDC->GetSafeHdc();                      // 获取设备上下文句柄
	CRect rect;
	// 矩形类
	GetDlgItem(IDC_ORIGIN)->GetClientRect(&rect); //获取box1客户区
	
	CvCapture *capture = cvCreateFileCapture("top_sample.avi");  //读取视频
	if (capture == NULL) {
		printf("NO capture");    //读取不成功，则标识
								 //return 1;
	};
	double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);   //读取视频的帧率
	int vfps = 1000 / fps;                                        //计算每帧播放的时间
	//printf("%5.1f\t%5d\n", fps, vfps);
	double frames = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);//读取视频中有多少帧
	//printf("frames is %f\n", frames);
	//cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //定义窗口
	IplImage *frame;
	CvvImage cimg;

//	int i = 0;
	while (1) {
		//for (int i = 0;i < 1000000;++i);
		frame = cvQueryFrame(capture);  
		if (!frame)break;                        //抓取帧
		cimg.CopyOf(frame, frame->nChannels);
		cimg.DrawToHDC(hdc, &rect);
		//KillTimer(TM_MSG);
		//SetTimer(TM_MSG, 100000000, NULL);
	//	float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
	//	printf("%f\n", ratio);
		
		//cvShowImage("IDC_STATIC",frame);                          //显示
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
	// TODO: 在此添加控件通知处理程序代码
	VideoCapture capture("top_sample.avi");
	if (capture.isOpened())	// 摄像头读取文件开关
	{
		while (true) //循环读取视频
		{
			//载入源图像并将其转换为灰度图
			capture >> g_srcImage;

			if (g_srcImage.empty()) //无法读取视频
			{
				printf(" No captured frame -- Break!");
				break;
			}

			framenum++;

			//尝试腐蚀操作
			Mat ele = getStructuringElement(MORPH_RECT, Size(5, 3));
			erode(g_srcImage, g_srcImage, ele);

			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);//转成灰度图

															  //中值滤波，去除无关的噪声
			medianBlur(g_grayImage, g_grayImage, median_thres);

			//阈值化，得到鱼躯干的轮廓图形
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);

			//提取躯干图形的轮廓
			vector<Vec4i> hierarchy;
			findContours(g_binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//Mat draw = Mat::zeros(g_binImage.size(), CV_8UC3);
			//Mat draw = g_binImage.clone();
			//cvtColor(draw, draw, COLOR_GRAY2BGR);
			//cout << "Area: ";
			double maxArea = 0;
			contour_num = 0;
			for (int i = 1; i < contours.size(); i++)//提取的轮廓中，除了图像外缘，最大的轮廓就是躯干
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


			//创建窗口，对图像进行角点识别
			if (maxArea >= AREA_LIMIT)//当得到的最大轮廓比较小的时候，不能保证算法的成立，会舍弃掉
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
