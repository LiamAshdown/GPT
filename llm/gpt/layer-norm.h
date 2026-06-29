#pragma once

#include <torch/torch.h>
#include "device.h"

static constexpr float LN_EPSILON = 1e-5f;

// https://arxiv.org/pdf/1607.06450
class LayerNorm {
public:
    LayerNorm(const int dims) {
        _gamma = torch::ones({ dims }, globalDevice());
        _beta = torch::zeros({ dims }, globalDevice());

        _derivativeGamma = torch::zeros({ dims }, globalDevice());
        _derivativeBeta  = torch::zeros({ dims }, globalDevice());
    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        params.push_back(_gamma);
        params.push_back(_beta);

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        grads.push_back(_derivativeGamma);
        grads.push_back(_derivativeBeta);

        return grads;
    }

    torch::Tensor forward(const torch::Tensor x) {
        _x = x;

        // Normalize the embedding values into a consistent scale
        // so we can't have one embedding with huge large numbers
        // and others with small numbers - this will mess up with our training
        // it's important the values remains at a consistent scale

        // reason why this happens because each block produces its embeddings
        // and if we don't normalize x after each block, our embeddings will explode in a large scale
        // which will cause imbalance with training
        const auto mean = x.mean(-1, true);
        const auto var = x.var(-1, false, true);

        _std = (var + LN_EPSILON).sqrt();
        _xNorm = (x - mean) / _std;

        return _gamma * _xNorm + _beta;
    }

    torch::Tensor backward(torch::Tensor dOut) {
        int N = dOut.size(-1);

        _derivativeGamma = (dOut * _xNorm).sum(0);
        _derivativeBeta = dOut.sum(0);

        auto dXNorm = dOut * _gamma;
        auto dVar = (dXNorm * (_x - _x.mean(-1, true)) * -0.5f
                     * _std.pow(-3)).sum(-1, true);
        auto dMean = (dXNorm * -1.0f / _std).sum(-1, true)
                     + dVar * (-2.0f / N) * (_x - _x.mean(-1, true)).sum(-1, true);

        return dXNorm / _std
               + dVar * 2.0f * (_x - _x.mean(-1, true)) / N
               + dMean / N;
    }

private:
    // Gamma and Beta is used to nudge the training
    torch::Tensor _gamma;
    torch::Tensor _beta;
    torch::Tensor _std;
    torch::Tensor _xNorm;

    torch::Tensor _x;

    torch::Tensor _derivativeGamma;
    torch::Tensor _derivativeBeta;
};