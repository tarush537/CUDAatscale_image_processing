# Blur Images

This template is part of the course project for the "CUDA at Scale for the Enterprise" course.

## Project Description
The project involves blurring multiple images, each sized 512 by 512 pixels. It reads unblurred images from one folder and outputs the blurred versions to another folder.

This is my submission for an assignment in the GPU Programming Specialization Coursera course from Johns Hopkins University.

### Installation and Usage

To install dependencies, run:
```shell
make install
```

To download the test dataset, run:
```shell
make data
```

To build the executable, run:
```shell
make build
```
This will produce `./bin/main.exe`.

To run the executable with default arguments, use:
```shell 
make run
```

For help with running `./bin/main.exe` with different arguments, execute it without any additional arguments.

To clean the built executable, run:
```shell 
make clean
```

To generate a zip file proving that the code ran correctly, run:
```shell 
make proof
```

## Code Organization

- **`bin/`**: Contains `main.exe` after running `make build`.
- **`data/`**: After running `make data`, this folder will contain textures in `./data/textures`. Running `main.exe` will place blurred texture files in `./data/output` using `make run`.
- **`lib/`**: No libraries are included; everything is installed using `sudo apt-get install`.
- **`src/`**: Contains all source files, including `main.cuh` (header file) and `main.cu` (code file).
- **`README.md`**: Describes the project to help others decide whether to clone the repository.
- **`INSTALL`**: Instructions for installing `libopencv` and `libboost` on Ubuntu:
  ```shell
  make install
  ```
- **`Makefile` or `CMakeLists.txt` or `build.sh`**: Contains the make rules.
- **`run.sh`**: Not included. Use `main.exe` in the `bin` folder instead.
