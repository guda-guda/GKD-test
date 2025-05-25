#include <winsock2.h>
#include <ws2tcpip.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstddef>
#include <string>
#include "json.hpp"
#include "Matrix.h"
#include "model.h"

using namespace cv;

template<typename T>
void sendMatrix(SOCKET client,const Matrix<T>& target)
{
    size_t rows = target.get_rows();
    size_t cols = target.get_colums();

    send(client,reinterpret_cast<const char*>(&rows),sizeof(rows),0);
    send(client,reinterpret_cast<const char*>(&cols),sizeof(cols),0);

    //发送整个矩阵
    send(client,reinterpret_cast<const char*>(target.elements.data()),rows*cols*sizeof(T),0);
}

template<typename T>
Matrix<T> recevMatrix(SOCKET client){
    //接受矩阵长宽
    size_t rows,cols;
    recv(client,reinterpret_cast<char*>(&rows),sizeof(rows),0);
    recv(client,reinterpret_cast<char*>(&cols),sizeof(cols),0);
    
    Matrix<T> result(rows,cols);
    
    //接收矩阵
    recv(client,reinterpret_cast<char*>(result.elements.data()),rows*cols*sizeof(T),0);

    return result;
}

int main()
{
    WSADATA WSAData;
    if(WSAStartup(MAKEWORD(2,2),&WSAData)!=0){
        std::cerr <<"WSAStartup failed." <<std::endl;
        return 1;
    }

    //创建套接字
    SOCKET client = socket(AF_INET,SOCK_STREAM,0);
    if(client == INVALID_SOCKET){
        std::cerr <<"创建套接字失败" <<std::endl;
        WSACleanup();
        return 1;
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
        return 1; 
    }

    Mat num = imread("E:\\code\\GKD-test\\Final version\\nums\\2.png",IMREAD_GRAYSCALE);
    if(num.empty())
    {
        std::cerr <<"无法加载图像" <<std::endl;
        return -1;
    }
    
    //调整大小
    Mat renum;
    resize(num,renum,Size(28,28),0,0,INTER_AREA);//调用resize方法调整图片矩阵大小
    
    //矩阵转化
    Matrix<float> mtxnum(renum);
    sendMatrix<float>(client,mtxnum);
    Matrix<float> result=recevMatrix<float>(client);
    std::cout <<"The result of the standard model is:" <<std::endl;
    result.print();
    
    //关闭套接字
    closesocket(client);
    WSACleanup();
    return 0;
}