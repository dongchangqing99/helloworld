/*
功能：使用opencv库以及Dlib库
作者：董常青
版本：V1.0
版权：董常青所有
历史记录：
2019/11/26，开始搭建；
2019/11/27，开始整理；
2019/12/03,继续整理；
参考资料：
https://blog.csdn.net/wangxing233/article/details/51549880;	  主要参考
https://blog.csdn.net/zmdsjtu/article/details/52235056；
https://blog.csdn.net/weixin_41215479/article/details/85252942；
https://blog.csdn.net/zmdsjtu/article/details/53454071;
https://blog.csdn.net/zmdsjtu/article/details/52422847;
https://blog.csdn.net/zqckzqck/article/details/78979219;
https://blog.csdn.net/zqckzqck/article/details/79040443;
https://blog.csdn.net/u012792343/article/details/78427368,解决JPEG图像加载问题
https://blog.csdn.net/yiyuehuan/article/details/70667318，
https://blog.csdn.net/ngy321/article/details/89453581,出错的原因是p可能不在rect内（或者边界超出），加个判断：
https://blog.csdn.net/yubin1277408629/article/details/53561037,DLIB转OPENCV，
http://wwwbuild.net/BigDataDigest/258755.html，
https://blog.csdn.net/u011473714/article/details/89379497，

备注说明：
需要借助Dlib库和OpenCV库，比较麻烦的是环境搭建，测试的环境为OpenCV2.4.11+VS2012+Dlib18.18

*/

#include <iostream>		//自己添加的头文件
#include <Windows.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>		//源程序自带的头文件
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------


//人脸检测函数
void faceLandmarkDetection(dlib::array2d<rgb_pixel>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{     
	dlib::frontal_face_detector detector = get_frontal_face_detector();		//Dlib库中的人脸检测器类对象

	std::vector<dlib::rectangle> dets = detector(img);		//rectangle表示人脸边界框，获得人脸轮廓框架
	
	full_object_detection shape = sp(img, dets[0]);		//获得68个人脸关键点坐标

	for (int i = 0; i < shape.num_parts(); ++i)			//将关键点坐标存储到landmark变量中
	{
		float x=shape.part(i).x();
		float y=shape.part(i).y();	
		landmark.push_back(Point2f(x,y));		
	}
}


//将边界4个顶点和4个中点添加上,从而拼接的时候，形成背景三角形进行拼接
void addKeypoints(std::vector<Point2f>& points,Size imgSize)
{
	points.push_back(Point2f(1,1));
	points.push_back(Point2f(1,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,1));
	points.push_back(Point2f(1,imgSize.height/2));
	points.push_back(Point2f(imgSize.width/2,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,imgSize.height/2));
	points.push_back(Point2f(imgSize.width/2,1));
}


//计算融合图像的特征点坐标
void morpKeypoints(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,std::vector<Point2f>& pointsMorph, double alpha)
{
	for (int i = 0; i < points1.size(); i++)	//获得融合图的关键点坐标
	{
		float x, y;
		x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
		y = (1 - alpha) * points1[i].y + alpha * points2[i].y;
		pointsMorph.push_back(Point2f(x, y));
	}
}


//定义一个结构体，用于存储序号
struct correspondens{
	std::vector<int> index;
};


//融合核心算法程序
void delaunayTriangulation(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,std::vector<Point2f>& pointsMorph,double alpha,std::vector<correspondens>& delaunayTri,Size imgSize)
{
	morpKeypoints(points1,points2,pointsMorph,alpha);	//调用函数，获得融合后关键点的坐标
	Rect rect(0, 0, imgSize.width, imgSize.height);		//将rect区域标记为新图尺寸，防止计算的坐标点超出了图像尺寸，报错误
	//for(int i=0;i<pointsMorph.size();++i)			//测试输出
	//{
	//	cout<<"x="<<pointsMorph[i].x<<",y= "<<pointsMorph[i].y<<endl;
	//}

	int k=0;
	cv::Subdiv2D subdiv(rect);		//建立三角剖分点集
	for (std::vector<Point2f>::iterator it = pointsMorph.begin(); it != pointsMorph.end(); it++)	//将76个关键点赋值到三角剖分点集中
	{
		if((*it).x >= rect.x && (*it).y >= rect.y && (*it).x < rect.x + rect.width && (*it).y < rect.y + rect.height)	//若出现超出尺寸范围的错误点，则不赋值
		{
			subdiv.insert(*it);
		}
	}
	
	std::vector<Vec6f> triangleList;		//这里无法获得序号，该函数只是获得可以组成拼接三角形的顶点坐标
	subdiv.getTriangleList(triangleList);		//通过三角剖分函数，获得六元组，即一个三角形三个顶点的坐标(x,y)组成
	
	for (size_t i = 0; i < triangleList.size(); ++i)	//将76个，每个三角形的顶点坐标赋值
	{
		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back( Point2f(t[0], t[1]) );
		pt.push_back( Point2f(t[2], t[3]) );
		pt.push_back( Point2f(t[4], t[5]) );
		//cout<<"pt.size() is "<<pt.size()<<endl;
		
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))	//通过三角索引，将原始图、融合图、结果图一一对应起来
		{
			//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<endl;
			int count = 0;
			for (int j = 0; j < 3; ++j)
			{
				for (size_t k = 0; k < pointsMorph.size(); k++)
				{
					if (abs(pt[j].x - pointsMorph[k].x) < 1.0   &&  abs(pt[j].y - pointsMorph[k].y) < 1.0)	//将每次找到的这三个点的序号找出来
					{
						ind.index.push_back(k);
						count++;
					}
				}
			}
			if (count == 3)
			{
				delaunayTri.push_back(ind);		//获得这些碎片三角形的顶点坐标的序号，即在76个关键点中的序列号，每个为一组
		
			}
		}
	}	
}


//根据对应的三角形三个角点集合计算仿射矩阵，然后将这个三角形图像进行仿射变换投影过去
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> & srcTri, std::vector<Point2f> & dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);	//将原图经过仿射变换，成为新图
}


//计算仿射矩阵，拼接图像，说明，图像不仅仅是对三角形区域进行拼接，而是对于整个矩形区域均进行拼接
void morphTriangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, double alpha)
{
	Rect r = cv::boundingRect(t);	//将包含三个顶点的三角形区域框出来，返回包含覆盖输入的最小正矩形
	Rect r1 = cv::boundingRect(t1);
	Rect r2 = cv::boundingRect(t2);

	std::vector<Point2f> t1Rect, t2Rect, tRect;		//这部分需要输出看一下
	std::vector<Point> tRectInt;
	for (int i = 0; i < 3; ++i)
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y));

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);			//mask默认为获得的最小正矩形ROI区域
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);		//调用函数，对于凸多边形进行填充,三角形填充区域为白色

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);
	img2(r2).copyTo(img2Rect);
	
	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);		//调用函数，将原始图的三角形碎片投影到结果图像中
	applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

	Mat imgRect = (1.0 - alpha)*warpImage1 + alpha*warpImage2;		//加权计算结果图点像素值
	
	multiply(imgRect, mask, imgRect);		//这里进行乘法运算的目的，就是将非三角形区域的像素值点乘以0，防止多图干扰
	multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));		//把矩形补丁的矩形区域复制到输出图像中
	img(r) = img(r) + imgRect;
	
}




//图像融合程序，核心程序
void morp(Mat &img1, Mat &img2, Mat& imgMorph, double alpha, const std::vector<Point2f> &points1, const std::vector<Point2f> &points2, const std::vector<correspondens> &triangle)
{
	img1.convertTo(img1, CV_32F);		//转化为浮点型
	img2.convertTo(img2, CV_32F);

	std::vector<Point2f> points;
	morpKeypoints(points1,points2,points,alpha);		//调用函数，融合关键点，这里实际上与获得三角碎片点那里重复了
	
	int x, y, z;
	int count = 0;
	for (int i=0;i<triangle.size();++i)		//遍历对所有三角形碎片进行融合处理
	{
		correspondens corpd=triangle[i];
		x = corpd.index[0];		//将三角形的顶点赋值
		y = corpd.index[1];
		z = corpd.index[2];
		std::vector<Point2f> t1, t2, t;
		t1.push_back(points1[x]);		//将原图A，融合图B，结果图C的三角关键点坐标提取出来
		t1.push_back(points1[y]);
		t1.push_back(points1[z]);

		t2.push_back(points2[x]);
		t2.push_back(points2[y]);
		t2.push_back(points2[z]);

		t.push_back(points[x]);
		t.push_back(points[y]);
		t.push_back(points[z]);
		morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);	//调用函数，进行图像融合
	}
}


//主函数，从这里开始
int main(int argc, char** argv)
{  
 //-------------- 第一步：加载图像 --------------------------------------------       
	shape_predictor sp;			//加载模型，初始化人脸定位关键点的检测器sp
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	
	Mat img1o = imread("image_A.jpg");		//读取图像
	Mat img2ot = imread("image_B.jpeg");
	if(!img1o.data)
	{
		cout<<"第一张图像读取失败，查看是否路径正确！"<<endl;
	}
	else
	{
		cout<<"第一张图像顺利读取！"<<endl;
	}
	if(!img2ot.data)
	{
		cout<<"第二张图像读取失败，查看是否路径正确！"<<endl;
	}
	else
	{
		cout<<"第二张图像顺利读取！"<<endl;
	}
	Mat img2o;
	int imgw = img1o.rows;			//获得基本图的尺寸
	int imgh = img1o.cols;
	dlib::array2d<rgb_pixel> img1(imgw,imgh),img2(imgw,imgh);		//定义一个array2d图像数组
	cv::resize(img2ot,img2o,cv::Size(imgh,imgw),(0,0),(0,0),1);		//将融合图尺寸调整到与基本图一致

	for(int i=0;i<imgw;++i)			//将OpenCV的Mat类型图像数组转换成Dlib的array2d类型数组
	{
		for(int j=0;j<imgh;++j)
		{
			img1[i][j].blue = img1o.at<cv::Vec3b>(i,j)[0];
			img1[i][j].green = img1o.at<cv::Vec3b>(i,j)[1];
			img1[i][j].red = img1o.at<cv::Vec3b>(i,j)[2];
		}
	}
	for(int i=0;i<imgw;++i)
	{
		for(int j=0;j<imgh;++j)
		{
			img2[i][j].blue = img2o.at<cv::Vec3b>(i,j)[0];
			img2[i][j].green = img2o.at<cv::Vec3b>(i,j)[1];
			img2[i][j].red = img2o.at<cv::Vec3b>(i,j)[2];
		}
	}
	
//----------------- 第二步：基于Dlib库，获得人脸关键点 ---------------------------------------------
	std::vector<Point2f> landmarks1,landmarks2;
	faceLandmarkDetection(img1,sp,landmarks1);		//调用函数，获得图像检测到的人脸轮廓以及68个关键点
	faceLandmarkDetection(img2,sp,landmarks2);
	
//	addKeypoints(landmarks1,img1o.size());		//增加原图边角8个点
//	addKeypoints(landmarks2,img2o.size());

	for (int i=0;i<landmarks1.size();++i)	//可以将找到的关键点，在原图中画圈显示出来
	{
		circle(img1o, landmarks1[i], 2, CV_RGB(255, 0, 0), 1, 8, 3);
	}
	namedWindow("landmark");
	imshow("landmark",img1o);
//	waitKey();

//--------------- 第三步：人脸融合 ----------------------------------------------
	std::vector<Mat> resultImage;		//存在多张图像时，将结果图存储在这个容器中
	resultImage.push_back(img1o);		//先将基本图存储
	for(double alpha= 0.25;alpha<1; alpha += 0.25)		//这里是按照0.25逐渐增加的比例进行融合
	{
		Mat imgMorph = Mat::zeros(img1o.size(), CV_32FC3);		//定义并初始化融合图像变量
		std::vector<Point2f> pointsMorph;

		std::vector<correspondens> delaunayTri;			//声明一个2D数组，用于存储三角碎片的顶点坐标数组
		delaunayTriangulation(landmarks1,landmarks2,pointsMorph,alpha,delaunayTri,img1o.size());	//调用函数，获得多个三角形碎片点的顶点序号以及坐标

		morp(img1o, img2o, imgMorph, alpha, landmarks1, landmarks2, delaunayTri);		//调用函数，图像融合
//		namedWindow("过程融合图像");
//		imshow("融合结果图像",imgMorph);		//显示融合结果图像
//		waitKey(0);

		resultImage.push_back(imgMorph);		//将融合图像存入数组
	}
	resultImage.push_back(img2o);		//将融合图下存储
	


//----------- 第四步：存储融合图像 --------------------------------
	
	for (int i=0;i<resultImage.size();++i)		//使用string字符串实现融合图像命名
	{
		string st="result";
		char t[20];
		sprintf(t, "%d", i);
		st=st+t;
		st=st+".jpg";
		imwrite(st,resultImage[i]);
	}
	cout<<"结束！"<<endl;
	system("pause");
	return 0;

}




