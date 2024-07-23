//
// Created by gaarg on 23-07-2024.
//

#include "../Include/main.h"
#include <opencv2/opencv.hpp>
#include "image_processing.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return -1;
    }

    std::string inputImagePath = argv[1];
    std::string outputImagePath = argv[2];

    cv::Mat inputImage = cv::imread(inputImagePath);
    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image: " << inputImagePath << std::endl;
        return -1;
    }

    cv::Mat outputImage;
    processImage(inputImage, outputImage);

    cv::imwrite(outputImagePath, outputImage);

    return 0;
}
