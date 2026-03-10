#include "../include/Model.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
using namespace std;

Model::Model(const vector<unsigned> &layerSizes) {
    unsigned numLayers = layerSizes.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        layers.push_back(Layer());
        unsigned numOutputs = layerNum == layerSizes.size() - 1 ? 0 : layerSizes[layerNum + 1];
        
        for (unsigned neuronNum = 0; neuronNum <= layerSizes[layerNum]; ++neuronNum) {
            layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        layers.back().back().setActivation(1.0);
    }
}

void Model::getPredictions(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
        resultVals.push_back(layers.back()[n].getActivation());
    }
}

void Model::backward(const vector<double> &targetVals) {
    Layer &outputLayer = layers.back();
    loss = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getActivation();
        loss += delta * delta;
    }
    loss /= outputLayer.size() - 1;
    loss = sqrt(loss);
    
    averageLoss = (averageLoss * smoothingFactor + loss) / (smoothingFactor + 1.0);

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calculateOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calculateHiddenGradients(nextLayer);
        }
    }
    
    for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateWeights(prevLayer);
        }
    }
}

void Model::forward(const vector<double> &inputVals) {
    assert(inputVals.size() == layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        layers[0][i].setActivation(inputVals[i]);
    }

    for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum) {
        Layer &prevLayer = layers[layerNum - 1];
        for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n) {
            layers[layerNum][n].forward(prevLayer);
        }
    }
}

void Model::saveWeights(const string &filename) const {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cout << "Error: Could not open file for saving weights" << endl;
        return;
    }
    for (unsigned layerNum = 0; layerNum < layers.size() - 1; ++layerNum) {
        const Layer &layer = layers[layerNum];
        for (unsigned n = 0; n < layer.size(); ++n) {
            const Neuron &neuron = layer[n];
            const vector<Connection> &weights = neuron.getWeights();
            for (unsigned c = 0; c < weights.size(); ++c) {
                outFile << weights[c].weight << " ";
            }
            outFile << endl;
        }
    }
    outFile.close();
    cout << "Success: Weights saved to " << filename << endl;
}