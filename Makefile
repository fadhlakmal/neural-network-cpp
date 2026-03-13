CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
TARGET = train_nn
SRCS = src/main.cpp src/Model.cpp src/Linear.cpp src/Activations.cpp src/Loss.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(TARGET) src/*.o weights.txt
