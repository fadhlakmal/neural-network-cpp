#include "../include/Neuron.hpp"
#include <cmath>
#include <cstdlib>

double Neuron::learningRate = 0.15;
double Neuron::momentum = 0.5;

double Neuron::randomWeight() { return rand() / double(RAND_MAX); }
double Neuron::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double Neuron::sigmoidDerivative(double x) { return (1.0 - x) * x; }

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }
    index = myIndex;
}

void Neuron::updateWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldWeightDelta = neuron.outputWeights[index].weightDelta;
        double newWeightDelta = learningRate * neuron.getActivation() * gradient + momentum * oldWeightDelta;
        
        neuron.outputWeights[index].weightDelta = newWeightDelta;
        neuron.outputWeights[index].weight += newWeightDelta;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }
    return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::sigmoidDerivative(activation);
}

void Neuron::calculateOutputGradients(double targetVal) {
    double delta = targetVal - activation;
    gradient = delta * Neuron::sigmoidDerivative(activation);
}

void Neuron::forward(const Layer &prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getActivation() * prevLayer[n].outputWeights[index].weight;
    }
    activation = Neuron::sigmoid(sum);
}