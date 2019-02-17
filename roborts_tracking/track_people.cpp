#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include <math.h>
#include <ros/ros.h>
#include <ros/package.h>
#include "geometry_msgs/Twist.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <cxcore.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <new>
#include "KCFcpp/src/kcftracker.hpp"
using namespace cv;
using namespace std;
static const std::string RGB_WINDOW = "RGB Image window";

float linear_speed = 0;
float rotation_speed = 0;

// 单位像素宽/高(cm/pixel)
#define UNIT_PIXEL_W 0.0008234375
#define UNIT_PIXEL_H 0.000825
cv::Rect roi;
int roi_width = 80;
int roi_height = 150;
int img_width = 640;
int img_height = 480;
int yawRate=0;
cv::Mat frame;
//HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1,-1, cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
cv::Rect selectRect;
cv::Point origin;
bool select_flag = false;
bool bRenewROI = false; // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bBeginKCF = false;

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool LAB = false;
bool has_dectect_people = false;
// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);



     /*
    SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
    将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个行向量，将该向量前面乘以-1。之后，再该行向量的最后添加一个元素rho。
    如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
    */
void preparePeopleDetect()
{
        ROS_INFO("SUCCESSFUL!!!!");
        has_dectect_people = false;
        int DescriptorDim;
       // hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("/home/kong/roborts_ws/src/RoboRTS/roborts_tracking/KCFcpp/src/SVM_HOG_2400PosINRIA_12000Neg_HardExample(误报少了漏检多了).xml");

        if(svm->empty())
        {
            cout<<"load svm detector failed!!!!"<<endl;
            return;
        }
        Mat svecsmat = svm->getSupportVectors();
        //特徵向量維數
        int svdim = svm->getVarCount();
        int numofsv = svecsmat.rows;
        //alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
        Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//每个支持向量对应的参数(拉格郎日乘子)
        Mat svindex = Mat::zeros(1, numofsv, CV_64F);//支持向量所在的索引

        Mat Result;
        double rho = svm->getDecisionFunction(0, alphamat, svindex);
        //将alphamat元素的数据类型重新转成CV_32F
        alphamat.convertTo(alphamat, CV_32F);
        Result = -1 * alphamat * svecsmat;

        vector<float> vec;
        for(int i=0;i<svdim;++i)
        {
            vec.push_back(Result.at<float>(0,i));
        }

        vec.push_back(rho);

        //saving HOGDetectorForOpenCV.txt
        ofstream fout("HOGDetectorForOpenCV.txt");
        for (int i = 0; i < vec.size(); ++i)
        {
            fout << vec[i] << endl;
        }

        hog.setSVMDetector(vec);
        printf("Start the tracking process\n");
} //行人检测
void peopleDetect()
{
        if (has_dectect_people)
            return;
        vector<Rect> found, found_filtered;
        double t = (double) getTickCount();
        //多尺度檢測
        hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
        t = (double) getTickCount() - t;
        printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
        size_t i, j;
        printf("found.size==%d", found.size());
        //去掉空間中具有內外包含關系的區域 保留大的
        for (i = 0; i < found.size(); i++) {
            Rect r = found[i];
            for (j = 0; j < found.size(); j++)
                if (j != i && (r & found[j]) == r)
                    break;
            if (j == found.size())
                found_filtered.push_back(r);
        }
        //適當縮小矩形
        Rect r;
        for (i = 0; i < found_filtered.size(); i++) {
            r = found_filtered[i]; // the HOG detector returns slightly larger rectangles than the real objects.
            // so we slightly shrink the rectangles to get a nicer output.
            r.x += cvRound(r.width * 0.1);
            r.width = cvRound(r.width * 0.8);
            r.y += cvRound(r.height * 0.07);
            r.height = cvRound(r.height * 0.8);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
            //printf("r.x==%d,y==%d,width==%d,height==%d\n",r.x,r.y,r.width,r.height);
            //cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2, 8, 0);

        }
        if (r.width > 100 && r.height > 300) {
            ROS_INFO("has dectect_people");
            has_dectect_people = true;
            selectRect.x = r.x + (r.width - roi_width) / 2;
            selectRect.y = r.y + (r.height - roi_height) / 2;
            selectRect.width = roi_width;
            selectRect.height = roi_height;
            printf("selectRect.x==%d,y==%d,width==%d,height==%d\n", selectRect.x, selectRect.y,
                   selectRect.width, selectRect.height);
        }
        //imshow(RGB_WINDOW, frame);
}
void imageCb(cv::Mat frame) {
    int dx=0;
    int dy=0;

    int k = 1920/img_width;
    peopleDetect();
    if (has_dectect_people && !select_flag)
    {
        printf("has_dectect_people = true \n");
        selectRect &= cv::Rect(0, 0, frame.cols, frame.rows);
        bRenewROI = true;
        select_flag = true;
    }
    if (bRenewROI)
    {
        tracker.init(selectRect, frame);
        bBeginKCF = true;
        bRenewROI = false;
    }
    if (bBeginKCF)
    {
        roi = tracker.update(frame);
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 1, 8);
        dx = (int)(roi.x + roi.width/2  - img_width/2);
        cv::circle(frame, Point(img_width/2, img_height/2), 5, cv::Scalar(255,0,0), 2, 8);
        if(roi.width != 0)
        {
            cv::circle(frame, Point(roi.x + roi.width/2, roi.y + roi.height/2), 3, cv::Scalar(0,0,255), 1, 8);

            cv::line(frame,  Point(img_width/2, img_height/2),
                     Point(roi.x + roi.width/2, roi.y + roi.height/2),
                     cv::Scalar(0,255,255));
        }
        yawRate = -dx;
        if(abs(yawRate) < 10/k)
        {
            yawRate = 0;
        }
        else if(abs(yawRate)>500/k) {
            yawRate = ((yawRate>0)?1:-1)*500/k;
        }
        rotation_speed = yawRate/180.*M_PI/160.*k;



    }
    else
        cv::rectangle(frame, selectRect, cv::Scalar(0, 255, 0), 2, 8, 0);
    cv::imshow(RGB_WINDOW, frame);
    cv::waitKey(1);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "kcf_tracker");
    ros::NodeHandle nh;
    ros::Publisher pub;
    pub = nh.advertise<geometry_msgs::Twist>("out_image_base_topic", 1000);
    preparePeopleDetect();
    cv::namedWindow(RGB_WINDOW);
    cv::VideoCapture video(0);

    video.set(CV_CAP_PROP_FRAME_WIDTH,img_width);
    video.set(CV_CAP_PROP_FRAME_HEIGHT,img_height);
    img_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    img_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    if(!video.isOpened())
    {
        ROS_INFO("cannot read video!\n");
        return -1;
    }
    KCFTracker *tracker = NULL;
    while(ros::ok())
    {

        if(video.read(frame))
        {
            imageCb(frame);
        }
        ros::spinOnce();
        geometry_msgs::Twist twist;
        twist.linear.x = linear_speed;
        twist.linear.y = 0;
        twist.linear.z = 0;
        twist.angular.x = 0;
        twist.angular.y = 0;
        twist.angular.z = rotation_speed;
        pub.publish(twist);
        if (cvWaitKey(33) == 'q')
            break;
    }
    cv::destroyWindow(RGB_WINDOW);
    return 0;
}
