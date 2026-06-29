#pragma once

#include "linear.h"

class FeedForward {
public:
    FeedForward(const int dims)
        : _fullyConnectExpand(dims, dims * 4),
        _fullyConnectShrink(dims * 4, dims) {
    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        auto ep = _fullyConnectExpand.parameters();
        params.insert(params.end(), ep.begin(), ep.end());

        auto sp = _fullyConnectShrink.parameters();
        params.insert(params.end(), sp.begin(), sp.end());

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        auto eg = _fullyConnectExpand.gradients();
        grads.insert(grads.end(), eg.begin(), eg.end());

        auto sg = _fullyConnectShrink.gradients();
        grads.insert(grads.end(), sg.begin(), sg.end());

        return grads;
    }

    torch::Tensor forward(const torch::Tensor &x) {
        _gulu = _fullyConnectExpand.forward(x);

        return _fullyConnectShrink.forward(torch::gelu(_gulu));
    }

    torch::Tensor backward(const torch::Tensor &dOut) {
        const auto derivativeGelu = _fullyConnectShrink.backward(dOut);
        const auto derivativePreGelu = derivativeGelu * geluGrad(_gulu);

        return _fullyConnectExpand.backward(derivativePreGelu);
    }

private:
    static torch::Tensor geluGrad(torch::Tensor x) {
        const float k = std::sqrt(2.0f / M_PI);
        const float c = 0.044715f;
        auto t = torch::tanh(k * (x + c * x.pow(3)));
        auto dt = 1.0f - t.pow(2);
        return 0.5f * (1.0f + t) + 0.5f * x * dt * k * (1.0f + 3.0f * c * x.pow(2));
    }

private:
    Linear _fullyConnectExpand;
    Linear _fullyConnectShrink;

    torch::Tensor _gulu;
};