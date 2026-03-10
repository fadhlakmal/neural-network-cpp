#pragma once
#include <vector>
using namespace std;

struct Connection {
    double weight;
    double weightDelta;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron {
private:
    static double learningRate;
    static double momentum;
    
    static double sigmoid(double x); 
    static double sigmoidDerivative(double x); 
    static double randomWeight();
    
    double sumDOW(const Layer &nextLayer) const;
    
    double activation;
    vector<Connection> outputWeights;
    unsigned index;
    double gradient;

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    const vector<Connection>& getWeights() const { return outputWeights; }
    void setActivation(double val) { activation = val; }
    double getActivation(void) const { return activation; }
    
    void forward(const Layer &prevLayer);
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer &nextLayer);
    void updateWeights(Layer &prevLayer);
};