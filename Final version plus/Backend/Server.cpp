#include <winsock2.h>
#include <ws2tcpip.h>
#include "Matrix.h"
#include "model.h"
#include <iostream>

template<typename T>
Matrix<T> recvMatrix(SOCKET client)
{   
    size_t rows,cols; 
    recv(client,reinterpret_cast<char*>(&rows),sizeof(rows),0);
    recv(client,reinterpret_cast<char*>(&cols),sizeof(cols),0);
    
    Matrix<T> recvMax(rows,cols);

    //接收客户端传递的矩阵
    recv(client,reinterpret_cast<char*>(recvMax.elements.data()),rows*cols*sizeof(T),0);

    return recvMax;
}

template<typename T>
void sendMatrix(SOCKET client,const Matrix<T>& result)
{
    size_t rows = result.get_rows();
    size_t cols = result.get_colums();

    send(client,reinterpret_cast<const char*>(&rows),sizeof(rows),0);
    send(client,reinterpret_cast<const char*>(&cols),sizeof(cols),0);

    //发送整个矩阵
    send(client,reinterpret_cast<const char*>(result.elements.data()),rows*cols*sizeof(T),0);
}

template<typename T>
void process()
{   
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
    }

    //创建套接字
    SOCKET serverSocket = socket(AF_INET,SOCK_STREAM,0);
    if(serverSocket == INVALID_SOCKET)
    {
        std::cerr <<"创建监听套接字失败" <<std::endl;
        WSACleanup();
    }
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(12345);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    //将套接字与地址绑定
    if(bind(serverSocket,reinterpret_cast<sockaddr*>(&serverAddr),sizeof(serverAddr))==SOCKET_ERROR){
        std::cerr <<"套接字绑定地址失败" <<std::endl;
        closesocket(serverSocket);
        WSACleanup();
    }

    //监听
    if(listen(serverSocket,1)==SOCKET_ERROR){
        std::cerr <<"监听失败" <<std::endl;
        closesocket(serverSocket);
        WSACleanup();
    }

    std::cout<<"服务器已启动，等待接收......" <<std::endl;

    //接收客户端连接
    while(true){
        sockaddr_in clientAddr;
        int clientLen = sizeof(clientAddr);
        SOCKET clientSocket = accept(serverSocket,reinterpret_cast<sockaddr*>(&clientAddr),&clientLen);

        if(clientSocket == INVALID_SOCKET){
            std::cerr <<"接受连接失败" <<std::endl;
            closesocket(serverSocket);
            closesocket(clientSocket);
            WSACleanup();
        }
        std::cout <<"客户端已连接" <<std::endl;

        //接受矩阵
        Matrix<T> recvMax=recvMatrix<T>(clientSocket);
        std::cout <<"已接收矩阵" <<std::endl;
        
        Basemodel<float>* m1 = createmodel<float>("E:\\code\\GKD-test\\part2\\mnist-fc");
        Matrix<T>result = m1->forward(recvMax);
        //发送结果
        sendMatrix(clientSocket,result);
        std::cout <<"结果已发送" <<std::endl;

        closesocket(clientSocket);
    }
}

int main()
{
    process<float>();
    return 0;
}