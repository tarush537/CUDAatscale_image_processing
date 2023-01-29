//
// Created by guidocalvano on 1/27/23.
//

#include "../include/main.cuh"
using namespace cv;

__global__ void blurKernel(uchar* pixels, int row_count, int column_count, int color_count)
{
  // Compute the relevant indices

  // The index of the target pixel that will be blurred
  int pixelIndex = blockIdx.x * column_count * 3 + threadIdx.x * 3;

  // The start and end index of the region of pixels relevant to the blur.
  int blurRange = 10;
  int startX = max(((int)threadIdx.x) - blurRange, 0);
  int endX = min(threadIdx.x + blurRange, column_count);

  int startY = max(((int)blockIdx.x) - blurRange, 0);
  int endY = min(blockIdx.x + blurRange, row_count);

  // The area of the blur, needed for averaging later
  int width = endX - startX;
  int height = endY - startY;
  int pixelCount = width * height;

  // Gather all pixel information and store it in a temporary variable, to prevent race conditions.
  unsigned int nextRGB[3] = {0, 0, 0};

  for(int x = startX; x < endX; ++x) {
    for(int y = startY; y < endY; ++y) {
    for(int c = 0; c < 3; ++c)
    {
        nextRGB[c] += pixels[x * 3 + y * blockDim.x * 3 + c];
    }}}

  // Wait for all threads to gather their information, to prevent race condition.
  __syncthreads();

  // Write the updated value of the pixel.
  for(int c = 0; c < 3; ++c)
      pixels[pixelIndex + c] = (uchar)(nextRGB[c] / pixelCount); //  ((uchar) nextRGB[c] + .5);

}

__host__ void gpuBlur(Mat& img)
{
  // Allocate memory and send data to gpu
  uchar* d_pixels;
  int byte_count = img.total() * img.elemSize();

  cudaMalloc((float**)&d_pixels, byte_count);
  cudaMemcpy(d_pixels, img.data, byte_count, cudaMemcpyHostToDevice);

  // Apply the blur kernel to the image in place
  blurKernel<<<img.rows, img.cols>>>(d_pixels, img.rows, img.cols, img.channels());

  // Move data out of the gpu, and deallocate memory.
  cudaDeviceSynchronize();
  cudaMemcpy(img.data, d_pixels, byte_count, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_pixels);
  cudaDeviceSynchronize();
}

void blurFiles(std::string inputPathName, std::string outputPathName)
{
  DIR* inputPath = opendir(inputPathName.c_str());
  dirent* dp;

  while ((dp = readdir(inputPath)) != NULL) {
    std::string imagePath = inputPathName + std::string(dp->d_name);

    if(!boost::algorithm::ends_with(imagePath, ".tiff")) continue;

    printf(imagePath.c_str());
    printf("\n");

    Mat img = imread(imagePath, IMREAD_COLOR);
    if(img.rows != 512) {
      printf("Image height is not 512: %i\n"
             "Skipping\n", img.rows);
      continue;
    }
    if(img.rows != 512) {
      printf("Image width is not 512: %i\n"
             "Skipping", img.cols);
      continue;
    }
    if(img.channels() != 3) {
      printf("Image is not color.\n"
             "Skipping\n");
      continue;
    }
    gpuBlur(img);
    imwrite((outputPathName + std::string(dp->d_name)).c_str(), img);
  }
  (void)closedir(inputPath);
}


int main(int argc, char *argv[])
{
  if(argc < 3) {
      printf("Syntax: main.exe inputpath outputpath");
      return 1;
  }

  cudaError_t err = cudaDeviceReset();

  blurFiles(argv[1], argv[2]);

  err = cudaDeviceReset();
}