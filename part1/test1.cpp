//这个文件用于测试part1内容

#include "Matrix.h"
#include "model.h"
#include <iostream>

int main(){
    Matrix<float> A(2,3);
    A(0, 0) =-1.0f; A(0, 1) = 2.0f; A(0, 2) =-3.0f;
    A(1, 0) = 4.0f; A(1, 1) =-5.0f; A(1, 2) = 6.0f;
    Matrix<float> B(2,3);
    B(0, 0) = 1.0f; B(0, 1) = 2.0f; B(0, 2) = 3.0f;
    B(1, 0) = 4.0f; B(1, 1) = 5.0f; B(1, 2) = 6.0f;
    Matrix<float> C(3, 2);
    C(0, 0) = 7.0f; C(0, 1) = 8.0f;
    C(1, 0) = 9.0f; C(1, 1) = 10.0f;
    C(2, 0) = 11.0f; C(2, 1) = 12.0f;
    Matrix<float> D(2,3);
    Matrix<float> E(2,2);
    //正常打印  
    std::cout <<"A:" <<std::endl ; 
    A.print();
    //RELU函数
    Matrix<float> H= RELU(A);
    std::cout <<"RELU H:" << std::endl;
    H.print();
   //矩阵加法
    D = A + B;
    std::cout <<"D:" <<std::endl ;
    D.print();
    //矩阵乘法
    E = B * C;
    std::cout <<"E:" <<std::endl ;
    E.print();
    //SoftMax函数
    Matrix<float> F(1,3);
    Matrix<float> G(3,1);
    F(0, 0) = 1.0f; F(0, 1) = 2.0f; F(0, 2) = 3.0f;
    G(0, 0) = 7.0f; G(1, 0) = 9.0f; G(2, 0) = 11.0f; 
    Matrix<float> result1 = softmax(F);
    Matrix<float> result2 = softmax(G);
    result1.print();
    result2.print();
    //model类的测试
    Matrix<float> w1(784, 500);
    Matrix<float> b1(1, 500);
    Matrix<float> w2(500, 10);
    Matrix<float> b2(1, 10);
    model test(w1,b1,w2,b2);
    Matrix<float> input(1, 784);
    Matrix<float> result3=test.forward(input);
    std::cout <<"The result of the standard model is:" <<std::endl;
    result3.print();
}