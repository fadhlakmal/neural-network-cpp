#include "../include/Conv2D.hpp"
#include <cstdlib>
#include <cmath>

Conv2D::Conv2D(int inC, int inH, int inW, int outC, int kSize) : inChannels(inC), outChannels(outC), inHeight(inH), inWidth(inW), kernelSize(kSize) {
    outHeight = inHeight - kernelSize + 1;
    outWidth = inWidth - kernelSize + 1;

    kernels.resize(outChannels, std::vector<std::vector<std::vector<double>>>(inChannels, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize))));
    biases.resize(outChannels, 0.0);

    for(int oc = 0; oc < outChannels; ++oc) {
        for(int ic = 0; ic < inChannels; ++ic) {
            for(int ky = 0; ky < kernelSize; ++ky) {
                for(int kx = 0; kx < kernelSize; ++kx) {
                    kernels[oc][ic][ky][kx] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / (inChannels * kernelSize * kernelSize));
                }
            }
        }
    }
}

std::vector<double> Conv2D::forward(const std::vector<double>& input) {
    lastInput = input;
    std::vector<double> output(outChannels * outHeight * outWidth, 0.0);

    for (int oc = 0; oc < outChannels; ++oc) {
        for (int oh = 0; oh < outHeight; ++oh) {
            for (int ow = 0; ow < outWidth; ++ow) {
                double sum = biases[oc];

                for (int ic = 0; ic < inChannels; ++ic) {
                    for (int ky = 0; ky < kernelSize; ++ky) {
                        for (int kx = 0; kx < kernelSize; ++kx) {
                            int ih = oh + ky;
                            int iw = ow + kx;
                            sum += input[getIndex(ic, ih, iw, inHeight, inWidth)] * kernels[oc][ic][ky][kx];
                        }
                    }
                }

                output[getIndex(oc, oh, ow, outHeight, outWidth)] = sum;
            }
        }
    }

    return output;
}

std::vector<double> Conv2D::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> inputGradient(lastInput.size(), 0.0);
    std::vector<std::vector<std::vector<std::vector<double>>>> kernelGradients(outChannels, std::vector<std::vector<std::vector<double>>>(inChannels, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize, 0.0))));

    for (int oc = 0; oc < outChannels; ++oc) {
        for (int oh = 0; oh < outHeight; ++oh) {
            for (int ow = 0; ow < outWidth; ++ow) {
                double grad = outputGradient[getIndex(oc, oh, ow, outHeight, outWidth)];

                for (int ic = 0; ic < inChannels; ++ic) {
                    for (int ky = 0; ky < kernelSize; ++ky) {
                        for (int kx = 0; kx < kernelSize; ++kx) {
                            int ih = oh + ky;
                            int iw = ow + kx;
                            kernelGradients[oc][ic][ky][kx] += lastInput[getIndex(ic, ih, iw, inHeight, inWidth)] * grad;
                            inputGradient[getIndex(ic, ih, iw, inHeight, inWidth)] += kernels[oc][ic][ky][kx] * grad; 
                        }
                    }
                }

                biases[oc] -= learningRate * grad;
            }
        }
    }

    for (int oc = 0; oc < outChannels; ++oc) {
        for (int ic = 0; ic < inChannels; ++ic) {
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    kernels[oc][ic][ky][kx] -= learningRate * kernelGradients[oc][ic][ky][kx];
                }
            }
        }
    }

    return inputGradient;
}
