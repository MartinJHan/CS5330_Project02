cmake_minimum_required(VERSION 3.15)
project(project2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(cbir_baseline
        src/cbir_baseline.cpp
        src/csv_util.cpp
)

target_link_libraries(cbir_baseline ${OpenCV_LIBS})