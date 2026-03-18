#include "../include/MaxPool2D.hpp"
#include <cstdlib>
#include <limits>

MaxPool2D::MaxPool2D(int inC, int inH, int inW, int poolS) : inChannels(inC), inHeight(inH), inWidth(inW), poolSize(poolS) {
    outHeight = inHeight / poolSize;
    outWidth = inWidth / poolSize;
    lastMaxIndices.resize(inChannels * outHeight * outWidth, -1);    
}

std::vector<double> MaxPool2D::forward(const std::vector<double>& input) {
    std::vector<double> output(inChannels * outHeight * outWidth, 0.0);

    for (int ic = 0; ic < inChannels; ++ic) {
        for (int oh = 0; oh < outHeight; ++oh) {
            for (int ow = 0; ow < outWidth; ++ow) {
                double maxVal = -std::numeric_limits<double>::infinity(); 
                int maxIdx = -1;

                for (int py = 0; py < poolSize; py++) {
                    for (int px = 0; px < poolSize; px++) {
                        int ih = oh * poolSize + py;
                        int iw = ow * poolSize + px;

                        if (input[getIndex(ic, ih, iw, inHeight, inWidth)] > maxVal) {
                            maxVal = input[getIndex(ic, ih, iw, inHeight, inWidth)];
                            maxIdx = getIndex(ic, ih, iw, inHeight, inWidth);
                        }
                    }
                }

                output[getIndex(ic, oh, ow, outHeight, outWidth)] = maxVal;
                lastMaxIndices[getIndex(ic, oh, ow, outHeight, outWidth)] = maxIdx;
            }
        }
    }

    return output;
}

std::vector<double> MaxPool2D::backward(const std::vector<double>& outputGradient, double learningRate) {
    (void)learningRate; 
    
    std::vector<double> inputGradient(inChannels * inHeight * inWidth, 0.0);

    for (size_t i = 0; i < outputGradient.size(); ++i) {
        int maxIdx = lastMaxIndices[i];
        if (maxIdx != -1) {
            inputGradient[maxIdx] += outputGradient[i];
        }
    }

    return inputGradient;
}
