# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(test VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加编译选项，确保输出编码为GBK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexec-charset=gbk")

# 包含头文件目录
include_directories(part2/include)

# 查找所有源文件
file(GLOB_RECURSE SOURCES "part2/*.cpp")

# 添加可执行文件
add_executable(test ${SOURCES})

# 查找并链接JSON库（假设使用的是nlohmann/json）
find_package(nlohmann_json REQUIRED)
target_link_libraries(test PRIVATE nlohmann_json::nlohmann_json)
