################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
# Makefile project only supported on Mac OS X, Linux, and Windows Platforms
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# Detect Windows
ifeq ($(OS),Windows_NT)
    HOST_OS := windows
    HOST_ARCH := $(shell uname -m 2>/dev/null || echo x86_64) # Default to x86_64
    TARGET_ARCH ?= $(HOST_ARCH)
    # For MinGW, the default is to use x86_64-w64-mingw32-g++
    HOST_COMPILER ?= x86_64-w64-mingw32-g++
    NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
else
    # architecture
    HOST_ARCH   := $(shell uname -m)
    TARGET_ARCH ?= $(HOST_ARCH)
    ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
        ifneq ($(TARGET_ARCH),$(HOST_ARCH))
            ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
                TARGET_SIZE := 64
            else ifneq (,$(filter $(TARGET_ARCH),armv7l))
                TARGET_SIZE := 32
            endif
        else
            TARGET_SIZE := $(shell getconf LONG_BIT)
        endif
    else
        $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
    endif
endif

# sbsa and aarch64 systems look similar. Need to differentiate them at host level for now.
ifeq ($(HOST_ARCH),aarch64)
    ifeq ($(CUDA_PATH)/targets/sbsa-linux,$(shell ls -1d $(CUDA_PATH)/targets/sbsa-linux 2>/dev/null))
        HOST_ARCH := sbsa
        TARGET_ARCH := sbsa
    endif
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-sbsa x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
ifeq ($(HOST_OS),)
    HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
endif
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android windows))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),sbsa)
        HOST_COMPILER ?= aarch64-linux-gnu-g++
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
else ifeq ($(TARGET_OS),qnx)
    LIBRARIES += -lsocket
    NVCCFLAGS += -Wno-deprecated-gpu-targets
endif
LIBRARIES  := -lcudart

# sample lists

CUDA_SAMPLES = \
        0_Simple/clock \
        0_Simple/matrixMul \
        0_Simple/simpleOccupancy \
        0_Simple/template \
        1_Utilities/bandwidthTest \
        1_Utilities/deviceQuery \
        1_Utilities/nbody \
        1_Utilities/p2pBandwidthLatencyTest \
        2_Graphics/Mandelbrot \
        3_Imaging/cudaDecodeGL \
        3_Imaging/cudaEncode \
        4_Finance/BlackScholes \
        4_Finance/binomialOptions \
        5_Simulations/conjugateGradient \
        5_Simulations/dwtHaar1D \
        6_Advanced/cdpBezierTessellation \
        6_Advanced/cdpQuadtree \
        6_Advanced/cdpSimplePrint \
        6_Advanced/cdpSimpleQuicksort \
        6_Advanced/simpleMPI \
        7_CUDALibraries/simpleCUFFT \
        7_CUDALibraries/simpleCUBLAS \
        7_CUDALibraries/simpleCUML \
        7_CUDALibraries/simpleCUSOLVER \
        7_CUDALibraries/simpleNPP \
        7_CUDALibraries/simpleNVRTC \
        7_CUDALibraries/simpleCUSPARSE \
        7_CUDALibraries/simpleThrust

# primary build targets
all: build

build: $(CUDA_SAMPLES)

run: $(addsuffix .run, $(CUDA_SAMPLES))

# cuda build rules
%: %.cu.o
	@echo "### $< ###"
	@$(HOST_COMPILER) $(CCFLAGS) $< -o $@ $(LDFLAGS) $(LIBRARIES)

%.cu.o: %.cu
	@$(NVCC) $(NVCCFLAGS) $< -c -o $@

# standard build rules
clean:
	rm -f $(addsuffix .o, $(CUDA_SAMPLES)) $(CUDA_SAMPLES)
	rm -f $(addsuffix .d, $(CUDA_SAMPLES))

clean_windows:
	del /f /q $(subst /,\,$(addsuffix .o, $(CUDA_SAMPLES))) $(subst /,\,$(CUDA_SAMPLES))
	del /f /q $(subst /,\,$(addsuffix .d, $(CUDA_SAMPLES)))

rebuild: clean build

rebuild_windows: clean_windows build

.PHONY: build run clean rebuild clean_windows rebuild_windows

