#pragma once
#include <vector>
#include <string>
#include "Neuron.hpp"
using namespace std;

class Model {
private:
    vector<Layer> layers;
    double loss;
    double averageLoss;
    double smoothingFactor;

public:
    Model(const vector<unsigned> &layerSizes);
    void forward(const vector<double> &inputVals);
    void backward(const vector<double> &targetVals);
    void getPredictions(vector<double> &resultVals) const;
    void saveWeights(const string &filename) const;
};