# Blur images
This is a filled in template for the course project for the CUDA at Scale for the Enterprise

## Project Description
The project blurs a number of images sized 512 by 512. It reads unblurred files from one folder and output blurred versions
in another folder.

It is my entry for an assignment in the gpu programming specialization coursera course from John Hopkins university.

To install dependencies run:
```shell
make install
```
To download the test dataset run:

```shell
make data
```

To build the executable run:
```shell
make build
```
This produces ./bin/main.exe.

To run this executable using default arguments run:
```shell 
make run
```

To get help on running ./bin/main.exe with different arguments, just run it without extra arguments.

To clean the built executable run:
```shell 
make clean
```

## Code Organization

```bin/```
This folder contains main.exe after running the following command:
```shell
make build
```

```data/```
After running the following command the data folder will contain textures in the folder ./data/textures.
```shell
make data
```
After running main.exe using the following command ./data/output will contain blurred texture files:
```shell
make run
```

```lib/```
There are no libraries. Everything is installed using sudo apt-get install.

```src/```
All source is in the src folder. It consists of a header file main.cuh and a code file main.cu.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
To install libopencv and libboost on ubuntu do:
```shell
make install
```

```Makefile or CMAkeLists.txt or build.sh```
Makefile contains the make rules.

```run.sh```
There is no run.sh. Just use main.exe in the bin folder.