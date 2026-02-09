cmake_minimum_required(VERSION 3.15)
project(project2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Executable for Tasks 1-4: Baseline matching with various features
add_executable(cbir_baseline
        src/cbir_baseline.cpp
        src/csv_util.cpp
)

target_link_libraries(cbir_baseline ${OpenCV_LIBS})

# Executable for Task 5: Deep Network Embeddings
add_executable(cbir_dnn
        src/cbir_dnn.cpp
)

target_link_libraries(cbir_dnn ${OpenCV_LIBS})

# Executable for Task 7: Custom Design (Banana Detector)
add_executable(cbir_custom
        src/cbir_custom.cpp
)

target_link_libraries(cbir_custom ${OpenCV_LIBS})