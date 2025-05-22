//本文件用于测试多线程乘法和单线程乘法所用时间
#include "model.h"
#include "Matrix.h"
#include <iostream>
#include <chrono>
#include <thread>

template<typename T>
void BlockMatrix_multiply(const Matrix<T>& m1,const Matrix<T>& m2,Matrix<T>& result,size_t startrow,size_t endrow)
{
    size_t col = m1.get_colums();
    size_t m2_cols =m2.get_colums();
    for(size_t i= startrow ; i<endrow;++i){
        for(size_t j = 0;j < m2_cols;++j){
            T outcome = 0;
            for(size_t k = 0;k < col ;++k){
                outcome += m1(i,k)*m2(k,j);  
            }
            result(i,j) = outcome;
        }
    }
}

template<typename T>
Matrix<T> Blockmultiply_threads(const Matrix<T>& m1,const Matrix<T>& m2,int core_use)
{   
    size_t r=m1.get_rows();
    size_t l=m2.get_colums();
    size_t startrow,endrow;
    std::vector<std::thread> threads;
    int numthreads = core_use;
    size_t blocksize = r/numthreads;

    Matrix<T> result(r,l);
    for(size_t i = 0;i<numthreads;++i)
    {
        startrow = i*blocksize;
        if(i == numthreads-1){
            endrow = r; //确保无论是否整除，都能包含所有行
        }
        else{
            endrow = blocksize*(i+1);
        }
        threads.emplace_back(BlockMatrix_multiply<T>,std::cref(m1),std::cref(m2),std::ref(result),startrow,endrow);
    }
    
    for(size_t j=0;j< numthreads;++j){
        threads[j].join();
    }
    return result;
}

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