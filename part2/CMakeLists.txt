# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(File-read VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加编译选项，确保输出编码为GBK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk")

# 添加静态库搜索路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 添加头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(File-read File-read.cpp)


# 链接静态库
target_link_libraries(File-read part1_static)






