#include <opencv2/opencv.hpp>
#include "image_processing.h"

__global__ void rgbToGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char r = input[3 * idx];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];
        output[idx] = 0.3f * r + 0.59f * g + 0.11f * b;
    }
}

void processImage(const cv::Mat& inputImage, cv::Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t grayImageSize = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, grayImageSize);
    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    unsigned char *h_output = new unsigned char[width * height];
    cudaMemcpy(h_output, d_output, grayImageSize, cudaMemcpyDeviceToHost);

    outputImage = cv::Mat(height, width, CV_8UC1, h_output).clone();

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
}
