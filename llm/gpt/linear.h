#pragma once

#include <torch/torch.h>
#include <cmath>
#include "device.h"

class Linear {
public:
    Linear(const int inDim, const int outDim) {
        // scale weights by 1/sqrt(inDim) to keep activations stable across layers
        _W = torch::randn({ inDim, outDim }, globalDevice()) * (1.0f / std::sqrt(static_cast<float>(inDim)));
        _bias = torch::zeros({ outDim }, globalDevice());

        _derivativeW = torch::zeros({ inDim, outDim }, globalDevice());
        _derivativeBias = torch::zeros({ outDim }, globalDevice());
    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        params.push_back(_W);
        params.push_back(_bias);

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        grads.push_back(_derivativeW);
        grads.push_back(_derivativeBias);

        return grads;
    }

    torch::Tensor forward(const torch::Tensor &x) {
        _x = x;

        return torch::matmul(x, _W) + _bias;
    }

    torch::Tensor backward(const torch::Tensor& derivativeOut) {
        _derivativeW = torch::matmul(_x.transpose(0, 1), derivativeOut);
        _derivativeBias = derivativeOut.sum(0);
        return torch::matmul(derivativeOut, _W.transpose(0, 1));
    }

private:
    torch::Tensor _x;

    torch::Tensor _W;
    torch::Tensor _bias;

    torch::Tensor _derivativeW;
    torch::Tensor _derivativeBias;

};
