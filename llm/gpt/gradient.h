#pragma once

#include <torch/torch.h>
#include <utility>
#include "device.h"

class Gradient {
public:
    static std::pair<torch::Tensor, torch::Tensor> crossEntropy(const torch::Tensor &logits,
                                                                const torch::Tensor &targets) {
        const int N = logits.size(0);

        // convert logits to probabilities
        const auto probs = torch::softmax(logits, -1);

        // loss: how confident was the model on the correct token
        const auto logProbs = torch::log_softmax(logits, -1);
        float lossVal = 0.0f;
        for (int i = 0; i < N; i++)
            lossVal += -logProbs[i][targets[i].item<int>()].item<float>();
        const auto loss = torch::tensor(lossVal / N, globalDevice());

        // find the target words specifically and -1 to let the adam optimizer to push it up or down
        // the other words which are not in the target words would be pushed down since they're wrong anyway
        // we need to know if our target word was correct or not by doing -1
        auto dLogits = probs.clone();
        for (int i = 0; i < N; i++)
            dLogits[i][targets[i].item<int>()] -= 1.0f;

        return {loss, dLogits / N};
    }
};
