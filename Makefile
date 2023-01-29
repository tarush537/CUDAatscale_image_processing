# IDIR=./
CXX = nvcc

#CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
#LDFLAGS += $(shell pkg-config --libs --static opencv)

.PHONY: all install data clean build run

all: data clean build run

#build:
#	$(CXX) ./src/main.cu --std c++17 `pkg-config opencv --cflags --libs` -o main.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -I./includes -lcuda

install:
	sudo apt-get install libopencv-dev
	sudo apt-get install libboost-all-dev

build:
	nvcc ./src/main.cu -o ./bin/main.exe -L/usr/local/cuda/lib64 -L /usr/lib/x86_64-linux-gnu/ -I/usr/local/cuda/include `pkg-config --cflags --libs opencv4` -ccbin g++-7

	# $(CXX) ./src/main.cu s--std c++17 `pkg-config opencv --cflags --libs` -o main.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -I./includes -lcuda


run:
	./bin/main.exe "./data/textures/" "./data/output/"

data:
	curl -s https://sipi.usc.edu/database/textures.tar.gz | tar xvz -C ./data

proof:
	tar -czvf proof.tar.gz ./data/output

clean:
	rm -f ./bin/main.exe

x11ide:  # just handy for accessing my machine over x11
	 tmux new-session -s ide "_JAVA_OPTIONS='-Dsun.java2d.xrender=false -Dsun.java2d.pmoffscreen=false' clion"
