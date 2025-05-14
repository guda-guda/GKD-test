
cmake_minimum_required(VERSION 4.0.0)

# set the project name 
project(test)

# 添加编译选项，确保在 add_executable 之前
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk")

# 指定头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 查找源文件
file(GLOB_RECURSE PART1_SOURCES "*.cpp")
# 排除测试文件
list(REMOVE_ITEM PART1_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

# 生成静态库
add_library(part1_static STATIC ${PART1_SOURCES})

# 如果还需要可执行的测试程序
add_executable(test test.cpp)
# 链接静态库到测试程序
target_link_libraries(test part1_static)