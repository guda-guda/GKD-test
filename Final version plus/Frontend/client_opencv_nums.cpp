#include <winsock2.h>
#include <ws2tcpip.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstddef>
#include <string>
#include <algorithm>
#include <chrono>
#include <numeric>
#include "json.hpp"
#include "Matrix.h"
#include "model.h"

using namespace cv;

Mat drawingBoard = Mat::ones(400,400,CV_8UC1)*255;//创建白色画布
bool isDrawing = false;
Point lastpoint;
std::mutex mtx;
std::vector<float> probablities(10,0.0f);
int recognizeDigit = -1;
bool running =true;
const int PREDICTION_HISTORY_SIZE = 5; // 保留最近5次预测结果
std::deque<int> predictionHistory;     // 预测历史队列
std::mutex historyMtx;                 // 历史队列的互斥锁

//鼠标回调函数
static void onMouse(int event,int x,int y,int flags,void* userdata)
{
    if(event == EVENT_LBUTTONDOWN){
        isDrawing = true;
        lastpoint = Point(x,y);
    }
    else if(event == EVENT_LBUTTONUP){
        isDrawing = false;
    }
    else if(event == EVENT_MOUSEMOVE && isDrawing == true){
        Point pt(x,y);
        line(drawingBoard,lastpoint,pt,Scalar(0),10,LINE_AA);
        lastpoint = pt;
    }
}

//图像处理以及识别函数
template<typename T>
void recognitionThread()
{
     WSADATA WSAData;
    if(WSAStartup(MAKEWORD(2,2),&WSAData)!=0){
        std::cerr <<"WSAStartup failed." <<std::endl;
        return;
    }

    //创建套接字
    SOCKET client = socket(AF_INET,SOCK_STREAM,0);
    if(client == INVALID_SOCKET){
        std::cerr <<"创建套接字失败" <<std::endl;
        WSACleanup();
        return;
    }

    //设置绑定服务器端口和IP
    sockaddr_in serverAddr;
    serverAddr.sin_family =AF_INET;
    serverAddr.sin_port = htons(12345);
    inet_pton(AF_INET,"127.0.0.1",&serverAddr.sin_addr);
    if(connect(client,reinterpret_cast<sockaddr*>(&serverAddr),sizeof(sockaddr))==SOCKET_ERROR){
        std::cerr <<"连接客户端失败...." <<std::endl;
        closesocket(client);
        WSACleanup();
        return; 
    }
    
    
    
    while(running){  
    Mat currentImage;
    {//复制当前画布用于识别
        std::lock_guard<std::mutex> lock(mtx);
        drawingBoard.copyTo(currentImage);
    }

    // 检查画布是否有内容（非白色像素）
    Scalar meanVal = mean(currentImage);
    if (meanVal[0] > 250) { // 如果图像几乎全白（未绘制）
    std::this_thread::sleep_for(std::chrono::milliseconds(500));   
    continue; // 跳过识别和发送
    }

    // 调试：保存当前发送的图像
    static int counter = 0;
    imwrite("send_" + std::to_string(counter++) + ".png", currentImage);

    //图像二值化预处理
    Mat processedImage;
    threshold(currentImage,processedImage,127,255,THRESH_BINARY_INV);

    //找到轮廓并提取最大轮廓作为数字
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;//存储轮廓层级关系
    findContours(processedImage,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    
    // 检查是否找到轮廓
    if (contours.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
    }

    //找到最大轮廓
    size_t largestCountourIdx = 0;
    double largestArea = 0;
    if(!contours.empty()){
        for(size_t i=0;i<contours.size();++i){
            double area = contourArea(contours[i]);
            if(area > largestArea){
                largestArea =area;
                largestCountourIdx = i;
            }
        }
    }

    //获取边界框
    Rect boundingBox = boundingRect(contours[largestCountourIdx]);

    //提取数字区域并调整为正方形
    int size = std::max(boundingBox.width,boundingBox.height);
    // 改进：增加20%边距
    int padding = size * 0.2;
    size += 2 * padding;    
    Mat digitROI = Mat::zeros(size,size,CV_8UC1);
    Mat roi = digitROI(Rect((size - boundingBox.width)/2+padding,
                             (size-boundingBox.height)/2+padding,
                            boundingBox.width,boundingBox.height));
    processedImage(boundingBox).copyTo(roi);

    //调整为28*28
    Mat resized;
    resize(digitROI,resized,Size(28,28),0,0,INTER_AREA);//调用resize方法调整图片矩阵大小
    resized.convertTo(resized, CV_32F, 1.0/255.0); 
    //矩阵转化
    Matrix<float> mtxnum(resized);
    // 发送矩阵数据
    size_t rows = mtxnum.get_rows();
    size_t cols = mtxnum.get_colums();

    send(client, reinterpret_cast<const char*>(&rows), sizeof(rows), 0);
    send(client, reinterpret_cast<const char*>(&cols), sizeof(cols), 0);
    send(client, reinterpret_cast<const char*>(mtxnum.elements.data()), rows*cols*sizeof(T), 0);

    if (send(client, reinterpret_cast<const char*>(&rows), sizeof(rows), 0) == SOCKET_ERROR ||
            send(client, reinterpret_cast<const char*>(&cols), sizeof(cols), 0) == SOCKET_ERROR ||
            send(client, reinterpret_cast<const char*>(mtxnum.elements.data()), rows * cols * sizeof(T), 0) == SOCKET_ERROR) {
            std::cerr << "发送数据失败" << std::endl;
            break;
        }

    // 接收识别结果
    size_t outRows, outCols;
    if (recv(client, reinterpret_cast<char*>(&outRows), sizeof(outRows), 0) <= 0 ||
            recv(client, reinterpret_cast<char*>(&outCols), sizeof(outCols), 0) <= 0) {
            std::cerr << "接收数据失败" << std::endl;
            break;
        }
        
    Matrix<T> output(outRows, outCols);
    if (recv(client, reinterpret_cast<char*>(output.elements.data()), outRows * outCols * sizeof(T), 0) <= 0) {
            std::cerr << "接收结果失败" << std::endl;
            break;;
        }

    
{   //更新概率和识别结果
    {
        std::lock_guard<std::mutex> lock(mtx);
        probablities.assign(output.elements.begin(), output.elements.end());
        recognizeDigit = distance(probablities.begin(),max_element(probablities.begin(),probablities.end()));
    }

     // 新增：更新预测历史
    {   std::lock_guard<std::mutex> histLock(historyMtx);
        predictionHistory.push_back(recognizeDigit);
        
        // 保持队列长度不超过设定值
        if (predictionHistory.size() > PREDICTION_HISTORY_SIZE) {
            predictionHistory.pop_front();
        }
    }
    // 调试输出
    std::cout << "接收概率: ";
    for (float p :probablities) std::cout << p << " ";
    std::cout << "\n预测数字: " << recognizeDigit << std::endl;
}
    
}
    //关闭套接字
    closesocket(client);
    WSACleanup();
}

// 新增函数：计算出现最频繁的数字
int getSmoothedPrediction() {
    std::lock_guard<std::mutex> lock(historyMtx);
    if (predictionHistory.empty()) return -1;

    // 统计频率
    std::unordered_map<int, int> freqMap;
    for (int num : predictionHistory) {
        freqMap[num]++;
    }

    // 找到最高频率的数字
    int maxFreq = 0;
    int result = predictionHistory.back(); // 默认返回最新结果
    for (const auto& pair : freqMap) {
        if (pair.second > maxFreq) {
            maxFreq = pair.second;
            result = pair.first;
        }
    }
    return result;
}

int main()
{   
    //创建窗口
    namedWindow("Draw a digit",WINDOW_NORMAL);
    resizeWindow("Draw a digit",800,400);

    //设置鼠标回调
    setMouseCallback("Draw a digit",onMouse,nullptr);

    //启用识别线程
    std::thread recognition(recognitionThread<float>);

    //主循环
    while(true){
        //创建显示图像
        Mat displayImage = Mat::zeros(400,800,CV_8UC3);

        //左侧：绘图区域
        Mat leftROI =displayImage(Rect(0,0,400,400));
        cvtColor(drawingBoard,leftROI,COLOR_GRAY2BGR);

        //右侧：概率柱状图和识别结果
        Mat rightROI = displayImage(Rect(400,0,400,400));
        rightROI = Scalar(0,0,0);

        //绘制结果
        {
            std::lock_guard<std::mutex> lock(mtx);
            //绘制识别的数字
            int recognizeDigit = getSmoothedPrediction();
            if(recognizeDigit != -1){
                putText(leftROI,"Prediction:",Point(50,50),
                        FONT_HERSHEY_SIMPLEX,0.7,Scalar(0,0,255),2);
                putText(leftROI,std::to_string(recognizeDigit),Point(250,50),
                        FONT_HERSHEY_SIMPLEX,1.5,Scalar(0,0,255),3);
            }
        }
        
        //绘制概率柱状图
        int barheight = 25;
        int barMargin = 10;
        int startY = 50;
        for(int i=0;i<10;i++){
            //计算当前柱子Y的位置
            int yPos =startY + i *(barheight + barMargin);

            //计算当柱子长度
            int barLength = static_cast<int>(probablities[i] * 250);

            //绘制数字标签
            putText(rightROI,std::to_string(i),Point(30,yPos+barheight-5),FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,255,0),1);

            //绘制水平柱状图
            rectangle(rightROI,Point(70,yPos),Point(70+barLength,yPos+barheight),Scalar(0,255-i*25,i*25),-1);

            //绘制百分比
            putText(rightROI,std::to_string(static_cast<int>(probablities[i] * 100))+"%",
                    Point(80 + barLength,yPos + barheight -5),FONT_HERSHEY_SIMPLEX,0.6,Scalar(0,255,0),1);
        }

        //显示图像
        imshow("Draw a digit",displayImage);
        waitKey(5); // 减少延迟至5ms

        //处理按键
        char key = waitKey(10);
        if(key == 27){
            //ESC键退出
            running =false;
            break;
        }
        else if(key == 'c' || key == 'C'){
            //清屏
            {
                std::lock_guard<std::mutex> lock(mtx);
                drawingBoard = Mat::ones(400,400,CV_8UC1) * 255;
                recognizeDigit =-1;
                fill(probablities.begin(),probablities.end(),0.0f);
            }
        }
    }

    //等待线程结束
    if(recognition.joinable()){
        recognition.join();
    }
    waitKey(0);
    return 0;
}