CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
TARGET = train_nn

$(TARGET): src/main.cpp src/Model.cpp src/Neuron.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/Model.cpp src/Neuron.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) weights.txt