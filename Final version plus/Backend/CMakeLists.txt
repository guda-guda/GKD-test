cmake_minimum_required(VERSION 4.0)

project(server)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项，确保输出编码为GBK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk") 

# 手动指定 OpenCV 库的搜索路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

# 指定 OpenCV 头文件路径
set(OpenCV_INCLUDE_DIRS "E:/OpenCV/build/x64/MinGW/include")
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# 添加可执行文件
add_executable(server Server.cpp)

# 链接OpenCV库到测试程序
target_link_libraries(server opencv_world4100)
target_link_libraries(server ws2_32)

# 使用 set_target_properties 设置可执行文件输出路径
set_target_properties(server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)    
