cmake_minimum_required(VERSION 3.10)
project(CUDA_Image_Processing)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture
set(CUDA_ARCHITECTURES "75")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set source files
set(SOURCE_FILES
        src/main.cpp
        src/Exceptions.cpp
        src/ImageIO.cpp
        src/ImagesCPU.cpp
        src/ImagesNPP.cpp
)

# Link against CUDA and other necessary libraries
cuda_add_executable(CUDA_Image_Processing ${SOURCE_FILES})
target_link_libraries(CUDA_Image_Processing ${CUDA_LIBRARIES} nppic nppif nppig nppial nppicc)

# Add any additional libraries or paths as needed
