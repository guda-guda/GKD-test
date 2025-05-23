#include <winsock2.h>
#include <ws2tcpip.h>
#include "Matrix.h"
#include "model.h"
#include <iostream>

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

int main(){
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

    Matrix<float> A(2,3);
    A(0, 0) =-1.0f; A(0, 1) = 2.0f; A(0, 2) =-3.0f;
    A(1, 0) = 4.0f; A(1, 1) =-5.0f; A(1, 2) = 6.0f;
    Matrix<float> result;

    //发送矩阵
    sendMatrix<float>(client,A);

    //接收结果
    result=recevMatrix<float>(client);

    //打印结果
    result.print();

    //关闭套接字
    closesocket(client);
    WSACleanup();
    return 0;
}