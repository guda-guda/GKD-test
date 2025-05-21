cmake_minimum_required(VERSION 4.0.0)

# set the project name 
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk")

# 手动指定 OpenCV 库的搜索路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 指定头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 指定 OpenCV 头文件路径
set(OpenCV_INCLUDE_DIRS "E:/OpenCV/build/x64/MinGW/include")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test test2.cpp)

# 链接OpenCV库到测试程序
target_link_libraries(test opencv_world4100)