//本文件用于测试模板类和多态
#include "model.h"
#include "Matrix.h"
#include <iostream>

int main(){
    Basemodel<float>* m1 = createmodel<float>("E:\\code\\GKD-test\\part2\\mnist-fc");
    Basemodel<double>* m2 = createmodel<double>("E:\\code\\GKD-test\\part2\\mnist-fc-plus");

    Matrix<float> input1(1, 784);
    Matrix<double> input2(1, 784);


    Matrix<float> result1 = m1->forward(input1);
    Matrix<double> result2 = m2->forward(input2);

    std::cout <<"The result1 of the standard model is:" <<std::endl;
    result1.print();
    std::cout <<"The result2 of the standard model is:" <<std::endl;
    result2.print();

    return 0;
}