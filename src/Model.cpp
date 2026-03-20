#include "../include/Model.hpp"
#include <fstream>
#include <stdexcept>

Model::~Model() { 
    // for(auto layer : layers) delete layer; 
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

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");
    for (Layer* layer : layers) {
        layer->save_weights(file);
    }
    file.close();
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");
    for (Layer* layer : layers) {
        layer->load_weights(file);
    }
    file.close();
}