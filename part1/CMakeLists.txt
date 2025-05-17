cmake_minimum_required(VERSION 4.0.0)

# set the project name 
project(test)

# 添加编译选项，确保在 add_executable 之前
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk")

# 手动指定 OpenCV 库的搜索路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 指定头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 指定 OpenCV 头文件路径
set(OpenCV_INCLUDE_DIRS "E:/OpenCV/build/x64/MinGW/include")
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找源文件
file(GLOB_RECURSE PART1_SOURCES "*.cpp")
# 排除测试文件
list(REMOVE_ITEM PART1_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

# 生成静态库
add_library(part1_static STATIC ${PART1_SOURCES})

# 链接OpenCV库到静态库
target_link_libraries(part1_static ${OpenCV_LIBS})

# 如果还需要可执行的测试程序
add_executable(test test.cpp)
# 链接静态库和OpenCV库到测试程序
target_link_libraries(test part1_static ${OpenCV_LIBS})