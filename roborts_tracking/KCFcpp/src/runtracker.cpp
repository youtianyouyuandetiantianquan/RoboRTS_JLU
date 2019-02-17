#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

    if (argc > 5) return -1;            // 输入大于5个参数

    bool HOG = true;                    // 是否使用hog特征
    bool FIXEDWINDOW = false;           // 是否使用修正窗口
    bool MULTISCALE = true;             // 是否使用多尺度
    bool SILENT = true;                 // 是否不做显示
    bool LAB = false;                   // 是否使用LAB颜色


    for(int i = 0; i < argc; i++){
        if ( strcmp (argv[i], "hog") == 0 )
            HOG = true;
        if ( strcmp (argv[i], "fixed_window") == 0 )
            FIXEDWINDOW = true;
        if ( strcmp (argv[i], "singlescale") == 0 )
            MULTISCALE = false;
        if ( strcmp (argv[i], "show") == 0 )
            SILENT = false;
        if ( strcmp (argv[i], "lab") == 0 ){
            LAB = true;
            HOG = true;
        }
        if ( strcmp (argv[i], "gray") == 0 )
            HOG = false;
    }





    // 創建kcf追蹤器
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // 當前幀
    Mat frame;

    //追蹤結果目標框
    Rect result;

    // images.txt的路径，用于读取图像
    ifstream listFile;
    string fileName = "images.txt";
    listFile.open(fileName);

    // 读取第一帧的目标区域
    ifstream groundtruthFile;
    string groundtruth = "region.txt";
    groundtruthFile.open(groundtruth);
    string firstLine;
    getline(groundtruthFile, firstLine);
    groundtruthFile.close();

    istringstream ss(firstLine);

    // 从给定的第一帧目标框读入四个顶点的坐标
    float x1, y1, x2, y2, x3, y3, x4, y4;
    char ch;
    ss >> x1;
    ss >> ch;
    ss >> y1;
    ss >> ch;
    ss >> x2;
    ss >> ch;
    ss >> y2;
    ss >> ch;
    ss >> x3;
    ss >> ch;
    ss >> y3;
    ss >> ch;
    ss >> x4;
    ss >> ch;
    ss >> y4;

    // 使用四个顶点计算出目标框
    float xMin =  min(x1, min(x2, min(x3, x4)));
    float yMin =  min(y1, min(y2, min(y3, y4)));
    float width = max(x1, max(x2, max(x3, x4))) - xMin;
    float height = max(y1, max(y2, max(y3, y4))) - yMin;


    // 读图像
    ifstream listFramesFile;
    string listFrames = "images.txt";
    listFramesFile.open(listFrames);
    string frameName;


    // 将结果写入output.txt
    ofstream resultsFile;
    string resultsPath = "output.txt";
    resultsFile.open(resultsPath);

    // 帧号计数
    int nFrames = 0;


    while ( getline(listFramesFile, frameName) ){
        frameName = frameName;

        // 读取列表上面的帧
        frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

        // 使用第一帧和目标框来初始化跟踪器
        if (nFrames == 0) {
            tracker.init( Rect(xMin, yMin, width, height), frame );
            rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
            resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
        }
            // 更新当前帧的结果
        else{
            result = tracker.update(frame);
            rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
            resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
        }

        nFrames++;
        // 显示并保存
        if (!SILENT){
            imshow("Image", frame);
            waitKey(1);
        }
    }
    // 关闭文件
    resultsFile.close();

    listFile.close();

}
