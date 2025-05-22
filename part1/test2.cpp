//本文件用于测试多线程乘法和单线程乘法所用时间
#include "model.h"
#include "Matrix.h"
#include <iostream>
#include <chrono>
#include <thread>

int main(){
    Matrix<float> w1(7840, 1000);
    Matrix<float> b1(1000, 1000);
    Matrix<float> w2(1000, 1000);
    Matrix<float> b2(1000, 1000);
    model test(w1,b1,w2,b2);
    Matrix<float> input(1000, 7840);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> result1 = RELU(input * w1 + b1) * w2+b2;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "单线程执行用时: " << elapsed.count() << " 秒" << std::endl;
    //单线程执行用时: 86.3432 秒
    
    //CPU核心数：32
    auto start1 = std::chrono::high_resolution_clock::now();
    Matrix<float> result2 = Blockmultiply_threads(RELU(Blockmultiply_threads(input,w1,32)+b1),w2,32)+b2;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "多线程执行用时: " << elapsed1.count() << " 秒" << std::endl;
    //多线程执行用时: 3.83136 秒
    return 0;
}