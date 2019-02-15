#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include <math.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/Twist.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <cxcore.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "KCFcpp/src/kcftracker.hpp"

using namespace cv;
using namespace std;
static const std::string RGB_WINDOW = "RGB Image window";
//static const std::string DEPTH_WINDOW = "DEPTH Image window";

#define Max_linear_speed 1
#define Min_linear_speed 0.4
#define Min_distance 1.0
#define Max_distance   5.0
#define Max_rotation_speed 0.75

float linear_speed = 0;
float rotation_speed = 0;

float k_linear_speed = (Max_linear_speed - Min_linear_speed) / (Max_distance - Min_distance);
float h_linear_speed = Min_linear_speed - k_linear_speed * Min_distance;

float k_rotation_speed = 0.004;
float h_rotation_speed_left = 1.2;
float h_rotation_speed_right = 1.36;

float distance_scale = 1.0;
int ERROR_OFFSET_X_left1 = 100;
int ERROR_OFFSET_X_left2 = 300;
int ERROR_OFFSET_X_right1 = 340;
int ERROR_OFFSET_X_right2 = 600;
int roi_height = 100;
int roi_width = 100;
cv::Mat rgbimage;
cv::Mat depthimage;
cv::Rect selectRect;
cv::Point origin;
cv::Rect result;
cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
bool select_flag = false;
bool bRenewROI = false; // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bBeginKCF = false;
bool enable_get_depth = false;

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;
bool has_dectect_people;
// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

float dist_val[5];
class ImageConverter
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    HOGDescriptor hog;

public:
    ros::Publisher pub;
    ImageConverter()
    : it(nh)
    { // Subscrive to input video feed and publish output video feed
        image_sub_ = it.subscribe("/camera/image", 1, &ImageConverter::imageCb, this);
        depth_sub_ = it.subscribe("/camera/depth/image", 1, &ImageConverter::depthCb, this);
        pub = nh.advertise<geometry_msgs::Twist>("out_image_base_topic", 1000);
        //pub = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);

        preparePeopleDetect();
        cv::namedWindow(RGB_WINDOW);
        //cv::namedWindow(DEPTH_WINDOW);
    }

    ~ImageConverter() {
        cv::destroyWindow(RGB_WINDOW);
        //cv::destroyWindow(DEPTH_WINDOW);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr; //申明一個CvImagePtr
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        //轉化爲opencv格式就可以對圖像進行操作
        cv_ptr->image.copyTo(rgbimage);
        peopleDetect();
        if (has_dectect_people && !select_flag)
        {
            printf("has_dectect_people = true \n");
            selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
            bRenewROI = true;
            select_flag = true;
        }
        //cv::setMouseCallback(RGB_WINDOW, onMouse, 0);
        if (bRenewROI)
        {
//            if (selectRect.width <= 0 || selectRect.height <= 0) {
//                bRenewROI = false;
//                continue;
//            }
            tracker.init(selectRect, rgbimage);
            bBeginKCF = true;
            bRenewROI = false;
            enable_get_depth = false;
        }
        if (bBeginKCF)
        {
            result = tracker.update(rgbimage);
            cv::rectangle(rgbimage, result, cv::Scalar(0, 255, 0), 1, 8);
            enable_get_depth = true;
        }
        else
            cv::rectangle(rgbimage, selectRect, cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::imshow(RGB_WINDOW, rgbimage);
        cv::waitKey(1);
    }
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
        //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("/home/kong/roborts_ws/src/RoboRTS/roborts_tracking/KCFcpp/src/12000neg_2400pos.xml");
        if(svm->empty())
        {
            cout<<"load svm detector failed!!!!"<<endl;
            return;
        }
        DescriptorDim = svm->getVarCount();
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
        hog.detectMultiScale(rgbimage, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
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
            //rectangle(rgbimage, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
            // printf("r.x==%d,y==%d,width==%d,height==%d\n",r.x,r.y,r.width,r.height);
        }
        if (r.width > 100 && r.height > 350) {
            has_dectect_people = true;
            selectRect.x = r.x + (r.width - roi_width) / 2;
            selectRect.y = r.y + (r.height - roi_height) / 2;
            selectRect.width = roi_width;
            selectRect.height = roi_height;
            printf("selectRect.x==%d,y==%d,width==%d,height==%d\n", selectRect.x, selectRect.y,
                   selectRect.width, selectRect.height);
        }
        //imshow(RGB_WINDOW, rgbimage);
    }
    void depthCb(const sensor_msgs::ImageConstPtr &msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv_ptr->image.copyTo(depthimage);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.",
                      msg->encoding.c_str());
        }
        if (enable_get_depth) {
            dist_val[0] = depthimage.at<float>(result.y + result.height / 3,
                                               result.x + result.width / 3);
            dist_val[1] = depthimage.at<float>(result.y + result.height / 3,
                                               result.x + 2 * result.width / 3);
            dist_val[2] = depthimage.at<float>(result.y + 2 * result.height / 3,
                                               result.x + result.width / 3);
            dist_val[3] = depthimage.at<float>(result.y + 2 * result.height / 3,
                                               result.x + 2 * result.width / 3);
            dist_val[4] = depthimage.at<float>(result.y + result.height / 2,
                                               result.x + result.width / 2);
            float distance = 0;
            int num_depth_points = 5;
            for (int i = 0; i < 5; i++) {
                if (dist_val[i] > 0.4 && dist_val[i] < 10.0)
                    distance += dist_val[i];
                else
                    num_depth_points--;
            }
            distance /= num_depth_points;

            //calculate linear speed
            if (distance > Min_distance)
                linear_speed = distance * k_linear_speed + h_linear_speed;
            else if (distance <= Min_distance - 0.5) {
                linear_speed = 0;
                linear_speed =
                        -1 * ((Min_distance - 0.5) * k_linear_speed + h_linear_speed);
            } else {
                linear_speed = 0;
            }


            if (fabs(linear_speed) > Max_linear_speed)
                linear_speed = Max_linear_speed;

            //calculate rotation speed
            int center_x = result.x + result.width / 2;
            if (center_x < ERROR_OFFSET_X_left1) {
                printf("center_x <<<<<<<< ERROR_OFFSET_X_left1\n");
                rotation_speed = Max_rotation_speed / 5;
                has_dectect_people = false;
                enable_get_depth = false;
                select_flag = false;
                bBeginKCF = false;
            } else if (center_x > ERROR_OFFSET_X_left1 &&
                       center_x < ERROR_OFFSET_X_left2)
                rotation_speed = -k_rotation_speed * center_x + h_rotation_speed_left;
            else if (center_x > ERROR_OFFSET_X_right1 &&
                     center_x < ERROR_OFFSET_X_right2)
                rotation_speed = -k_rotation_speed * center_x + h_rotation_speed_right;
            else if (center_x > ERROR_OFFSET_X_right2) {
                printf("center_x >>>>>>>> ERROR_OFFSET_X_right2\n");
                rotation_speed = -Max_rotation_speed / 5;
                has_dectect_people = false;
                enable_get_depth = false;
                select_flag = false;
                bBeginKCF = false;
            } else
                rotation_speed = 0;
            std::cout <<  "linear_speed = " << linear_speed << "  rotation_speed = " << rotation_speed << std::endl;
            // std::cout <<  dist_val[0]  << " / " <<  dist_val[1] << " / " << dist_val[2] << " / " << dist_val[3] <<  " / " << dist_val[4] << std::endl;
            // std::cout <<  "distance = " << distance << std::endl;
        }
        //cv::imshow(DEPTH_WINDOW, depthimage);
        cv::waitKey(1);
    }
};
int main(int argc, char** argv)
{

    ros::init(argc, argv, "kcf_tracker");
    ImageConverter ic;

    KCFTracker *tracker = NULL;
    while(ros::ok())
    {

        ros::spinOnce();
        geometry_msgs::Twist twist;
        twist.linear.x = linear_speed;
        twist.linear.y = 0;
        twist.linear.z = 0;
        twist.angular.x = 0;
        twist.angular.y = 0;
        twist.angular.z = rotation_speed;
        ic.pub.publish(twist);
        if (cvWaitKey(33) == 'q')
            break;
    }
    return 0;
}
