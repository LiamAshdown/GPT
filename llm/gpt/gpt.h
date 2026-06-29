#pragma once

#include "../../util/event-dispatcher/event-dispatcher.h"
#include "../../util/event-dispatcher/events/events.h"

#include "window.h"
#include "embedding.h"
#include "model.h"
#include "adam.h"
#include "device.h"

#include <torch/torch.h>
#include <string>
#include <algorithm>


class GPT {
public:
    GPT(const int vocabSize, const int seqLen, const int dims, const int layers, const int heads, EventDispatcher* dispatcher = nullptr) :
        _window(seqLen),
        _model(vocabSize, seqLen, dims, layers, heads),
        _dispatcher(dispatcher) {
        _adam.init(_model.parameters());
    }

    std::vector<int> predict(std::vector<int> context, const int newTokens, const float temperature = 1.0f, const int topK = 40) {
        for (int i = 0; i < newTokens; i++) {
            const int start = std::max(0, static_cast<int>(context.size()) - _window.seqLen());
            const std::vector<int> window(context.begin() + start, context.end());
            const auto input = torch::tensor(window, torch::TensorOptions().dtype(torch::kLong).device(globalDevice()));

            const auto logits = _model.forward(input);
            const auto lastLogits = logits[-1];

            // top-k sampling — only consider the K most likely tokens
            // this prevents the model getting stuck on degenerate tokens like whitespace
            const auto topk       = torch::topk(lastLogits, topK);
            const auto topkLogits  = std::get<0>(topk);
            const auto topkIndices = std::get<1>(topk);

            const auto probs    = torch::softmax(topkLogits / temperature, -1);
            const auto sampled  = torch::multinomial(probs, 1).item<int>();
            const auto nextToken = topkIndices[sampled].item<int>();

            context.push_back(nextToken);
        }

        return context;
    }

    void save(const std::string& path) {
        auto params = _model.parameters();
        std::vector<torch::Tensor> tensors;
        for (auto& p : params) {
            tensors.push_back(p.get());
        }
        torch::save(tensors, path);
    }

    void load(const std::string& path) {
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, path);
        auto params = _model.parameters();
        for (size_t i = 0; i < params.size(); i++) {
            params[i].get().copy_(tensors[i].to(globalDevice()));
        }
    }

    void fit(const std::vector<int>& tokenIds, const int steps) {
        _adam.setSchedule(steps);

        for (int step = 0; step < steps; step++) {
            train(tokenIds);

            if (_dispatcher) {
                _dispatcher->emit<StepEvent>({step, steps});
            }
        }
    }

    void train(const std::vector<int>& tokenIds) {
        _model.zeroGrad();

        const auto batch = _window.sample(tokenIds);

        float batchLoss = 0.0f;

        for (const auto& itr : batch) {
            auto currentLoss = _model.loss(itr);
            batchLoss += currentLoss.item<float>();

            _model.backward(itr);
        }

        batchLoss /= batch.size();

        if (_dispatcher) {
            _dispatcher->emit<BatchLossEvent>({batchLoss});
        }

        _adam.update(_model.parameters(), _model.gradients());
    }

private:

    Window _window;
    Model _model;
    Adam _adam;

    EventDispatcher* _dispatcher;
};