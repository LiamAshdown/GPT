#pragma once

#include <torch/torch.h>
#include <vector>
#include <cmath>

static constexpr float ADAM_EPSILON = 1e-8f;

// The purpose of this is to nudge our weights during training
// by using the gradients from the backward pass which stores what weights went wrong
// I don't really completely understand the maths here, but I've just copied
// the code from these two source files:
// https://github.com/SermetPekin/microgradCpp/blob/main/include/adam.hpp
// https://github.dev/karpathy/llm.c/tree/master/llmc
class Adam {
public:
    Adam(const float lr = 3e-4f, const float beta1 = 0.9f, const float beta2 = 0.999f)
        : _lr(lr), _lrMin(lr * 0.1f), _beta1(beta1), _beta2(beta2), _step(0),
          _warmupSteps(0), _totalSteps(0) {
    }

    void setSchedule(const int totalSteps, const int warmupSteps = 200) {
        _totalSteps  = totalSteps;
        _warmupSteps = warmupSteps;
    }

    void init(const std::vector<std::reference_wrapper<torch::Tensor>> &params) {
        for (auto& p : params) {
            _m.push_back(torch::zeros_like(p.get()));
            _v.push_back(torch::zeros_like(p.get()));
        }
    }

    void update(
        const std::vector<std::reference_wrapper<torch::Tensor>> &params,
        const std::vector<std::reference_wrapper<torch::Tensor>> &grads
    ) {
        _step++;

        float currentLr = _lr;
        // https://github.dev/karpathy/llm.c/tree/master/llmc
        {
            if (_totalSteps > 0) {
                if (_step <= _warmupSteps) {
                    currentLr = _lr * static_cast<float>(_step) / static_cast<float>(std::max(_warmupSteps, 1));
                } else {
                    const float progress = static_cast<float>(_step - _warmupSteps)
                                         / static_cast<float>(_totalSteps - _warmupSteps);
                    const float clipped  = std::min(progress, 1.0f);
                    currentLr = _lrMin + 0.5f * (_lr - _lrMin) * (1.0f + std::cos(M_PI * clipped));
                }
            }
        }

        // in the early training steps we want to nudge as much as we can
        // gradually as the steps increase, this fades away as the bc1 and bc2
        // becomes smaller
        // since at the start of the training all the values are random values
        // so we need to do a large nudge to get us going - start with a big correction
        const float bc1 = 1.0f - std::pow(_beta1, _step);
        const float bc2 = 1.0f - std::pow(_beta2, _step);

        for (size_t i = 0; i < params.size(); i++) {
            auto& g = grads[i].get();
            auto& p = params[i].get();

            // smooth the gradient, keep 90% of the original and tune it by 10% with the gradient
            // the 90% is the beta1
            _m[i] = _beta1 * _m[i] + (1.0f - _beta1) * g;
            //                          ( 1.0f - 0.9 = 10%)
            // tracks how volatile the gradients are
            _v[i] = _beta2 * _v[i] + (1.0f - _beta2) * g.pow(2);

            // bias corrected
            auto mHat = _m[i] / bc1;
            auto vHat = _v[i] / bc2;

            // nudge the weight
            p -= currentLr * mHat / (vHat.sqrt() + ADAM_EPSILON);
        }
    }

private:
    float _lr;
    float _lrMin;
    float _beta1;
    float _beta2;
    int   _step;
    int   _warmupSteps;
    int   _totalSteps;

    std::vector<torch::Tensor> _m;
    std::vector<torch::Tensor> _v;
};
