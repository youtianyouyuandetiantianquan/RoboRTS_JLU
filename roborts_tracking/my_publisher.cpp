#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream> // for converting the command line parameter to integer

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/image", 1);
    cv::VideoCapture video(0); // Check if video device can be opened with the given index
//    int img_width = 640;
//    int img_height = 480;
//    video.set(CV_CAP_PROP_FRAME_WIDTH,img_width);
//    video.set(CV_CAP_PROP_FRAME_HEIGHT,img_height);
//    img_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
//    img_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);

    if(!video.isOpened())
    {
        ROS_INFO("cannot read video!\n");
        return -1;
    }
    cv::Mat frame;
    sensor_msgs::ImagePtr msg;
    ros::Rate loop_rate(5);
    while (nh.ok())
    {
        //cap >> frame; // Check if grabbed frame is actually full with some content
        if(video.read(frame))
        {
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            pub.publish(msg);
            //cv::Wait(1);
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}
