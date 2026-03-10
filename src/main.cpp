#include <iostream>
#include <vector>
#include "../include/Model.hpp"

int main() {
    vector<unsigned> layerSizes = {2, 4, 1}; 
    Model myModel(layerSizes);

    vector<vector<double>> inputData = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    vector<vector<double>> targetData = {
        {0.0}, 
        {1.0}, 
        {1.0}, 
        {0.0}  
    };

    int epochs = 2000;
    cout << "\ntraining" << endl;
    
    for (int i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < inputData.size(); ++j) {
            myModel.forward(inputData[j]);
            myModel.backward(targetData[j]);
        }
        
        if (i % 500 == 0) {
            cout << "Epoch " << i << " completed." << endl;
        }
    }

    cout << "\ntesting" << endl;
    vector<double> predictions;

    for (size_t i = 0; i < inputData.size(); ++i) {
        myModel.forward(inputData[i]);
        myModel.getPredictions(predictions);
        
        cout << "Input: [" << inputData[i][0] << ", " << inputData[i][1] << "] "
             << "Target: " << targetData[i][0] << " "
             << "Prediction: " << predictions[0] << endl;
    }

    myModel.saveWeights("weights.txt");

    return 0;
}