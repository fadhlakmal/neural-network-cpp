#include "../include/Model.hpp"

Model::~Model() { 
    for(auto layer : layers) delete layer; 
}

void Model::add(Layer* layer) {
    layers.push_back(layer);
}

std::vector<double> Model::forward(std::vector<double> input) {
    for (auto layer : layers) {
        input = layer->forward(input);
    }
    return input;
}

void Model::backward(std::vector<double> gradient, double learningRate) {
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient, learningRate);
    }
}
