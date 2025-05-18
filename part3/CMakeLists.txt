cmake_minimum_required(VERSION 4.0)

project(opencv_nums)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项，确保输出编码为GBK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk") 

# 定义本地库目录为项目根目录下的 lib 文件夹
set(LOCAL_LIB_DIR ${CMAKE_SOURCE_DIR}/lib) 

# 指定 OpenCV 头文件路径
set(OpenCV_INCLUDE_DIRS "E:/OpenCV/build/x64/MinGW/include")
include_directories(${OpenCV_INCLUDE_DIRS})

# 指定头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(opencv_nums opencv_nums.cpp)

# 链接自定义静态库
target_link_libraries(opencv_nums ${LOCAL_LIB_DIR}/libpart1_static.a)

# 链接 OpenCV 动态库（导入库）
target_link_libraries(opencv_nums ${LOCAL_LIB_DIR}/libopencv_world4100.dll.a)
