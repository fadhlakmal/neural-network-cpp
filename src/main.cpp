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
    myModel.add(new ReLU());
    myModel.add(new Linear(4, 3)); 

    vector<vector<double>> inputData = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    vector<vector<double>> targetData = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, 
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0} 
    };

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
            cout << "Epoch " << i << " | Loss: " << epochLoss << endl;
        }
    }

    cout << "\ntesting" << endl;
    for (size_t i = 0; i < inputData.size(); ++i) {
        vector<double> logits = myModel.forward(inputData[i]);
        
        cout << "Input: [" << inputData[i][0] << ", " << inputData[i][1] << "] | ";
        
        cout << "Target: [";
        for(double t : targetData[i]) cout << t << " ";
        cout << "] | ";

        cout << "Logits: [";
        for(double l : logits) cout << l << " ";
        cout << "]" << endl;
    }

    return 0;
}
