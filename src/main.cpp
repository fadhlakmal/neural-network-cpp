#include <iostream>
#include <vector>
#include "../include/Model.hpp"
#include "../include/Conv2D.hpp"
#include "../include/Linear.hpp"
#include "../include/Activations.hpp"
#include "../include/Loss.hpp"
#include "../include/Utils.hpp"

using namespace std;

vector<double> flattenImage(vector<vector<double>> img2d) {
    vector<double> flat;
    for (auto& row : img2d) {
        for (double val : row) flat.push_back(val);
    }
    return flat;
}

int main() {
    Model myModel;
    
    myModel.add(new Conv2D(1, 4, 4, 2, 3));
    myModel.add(new ReLU());
    myModel.add(new Linear(8, 3));

    vector<double> img0 = flattenImage({
        {0, 0, 0, 0},
        {1, 1, 1, 1},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    });

    vector<double> img1 = flattenImage({
        {0, 1, 0, 0},
        {0, 1, 0, 0},
        {0, 1, 0, 0},
        {0, 1, 0, 0}
    });

    vector<double> img2 = flattenImage({
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    });

    vector<vector<double>> inputData = {img0, img1, img2};
    vector<vector<double>> targetData = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    int epochs = 10000;
    double learningRate = 0.05; 
    
    cout << "\ntraining" << endl;
    
    for (int i = 0; i < epochs; ++i) {
        double epochLoss = 0.0;
        for (size_t j = 0; j < inputData.size(); ++j) {
            vector<double> logits = myModel.forward(inputData[j]);
            
            epochLoss += Loss::cross_entropy_loss(logits, targetData[j]);
            
            vector<double> gradient = Loss::cross_entropy_gradient(logits, targetData[j]);
            myModel.backward(gradient, learningRate);
        }
        epochLoss /= inputData.size();
        
        if (i % 2000 == 0) {
            cout << "Epoch " << i << " | Avg Loss: " << epochLoss << endl;
        }
    }

    cout << "\ntesting" << endl;
    for (size_t i = 0; i < inputData.size(); ++i) {
        vector<double> logits = myModel.forward(inputData[i]);
        vector<double> probs = Utils::softmax(logits); 
        
        auto max_it = std::max_element(probs.begin(), probs.end());
        int predicted_class = std::distance(probs.begin(), max_it);
        
        cout << "Item " << i << " : [";
        for (size_t c = 0; c < probs.size(); ++c) {
            cout << probs[c] << (c < probs.size() - 1 ? ", " : "");
        }
        cout << "] | Pred Class: " << predicted_class << endl;
    }

    return 0;
}
