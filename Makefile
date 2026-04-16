CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -fPIC
TARGET = train_nn

PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
PY_SUFFIX = $(shell python3-config --extension-suffix)

SRCS = src/Model.cpp src/Linear.cpp src/Conv2D.cpp src/Activations.cpp src/Loss.cpp src/Utils.cpp src/MaxPool2D.cpp src/RNN.cpp src/Embedding.cpp src/Attention.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS) src/main.o
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.o -o $(TARGET)

pybind: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(PYBIND_INCLUDES) src/bindings.cpp $(OBJS) -o fent$(PY_SUFFIX)

clean:
	rm -f $(TARGET) src/*.o *.so *.pyd weights.txt
