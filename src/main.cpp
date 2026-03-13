#include <iostream>
#include <vector>
#include "../include/Model.hpp"
#include "../include/Linear.hpp"
#include "../include/Activations.hpp"
#include "../include/Loss.hpp"

using namespace std;

int main() {
    Model myModel;
    
    myModel.add(new Linear(2, 4));
    myModel.add(new Sigmoid());
    myModel.add(new Linear(4, 1));
    myModel.add(new Sigmoid());

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

    int epochs = 10000;
    double learningRate = 0.5;
    
    cout << "\ntraining" << endl;
    
    for (int i = 0; i < epochs; ++i) {
        double epochLoss = 0.0;
        for (size_t j = 0; j < inputData.size(); ++j) {
            vector<double> prediction = myModel.forward(inputData[j]);
            epochLoss += Loss::mse_loss(prediction, targetData[j]);
            
            vector<double> gradient = Loss::mse_gradient(prediction, targetData[j]);
            myModel.backward(gradient, learningRate);
        }
        epochLoss /= inputData.size();
        
        if (i % 2000 == 0) {
            cout << "Epoch " << i << " | Loss: " << epochLoss << endl;
        }
    }

    cout << "\ntesting" << endl;
    for (size_t i = 0; i < inputData.size(); ++i) {
        vector<double> prediction = myModel.forward(inputData[i]);
        
        cout << "Input: [" << inputData[i][0] << ", " << inputData[i][1] << "] "
             << "Target: " << targetData[i][0] << " "
             << "Prediction: " << prediction[0] << endl;
    }

    return 0;
}
