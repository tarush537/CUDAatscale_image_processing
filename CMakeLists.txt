cmake_minimum_required(VERSION 3.10)
project(ImageProcessingCUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Set the CUDA NVCC flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-lineinfo;-std=c++14)

# List of source files
set(SOURCES
        src/main.cpp
        src/image_processing.cu
)

# Add the executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
