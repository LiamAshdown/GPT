#pragma once

#include "block.h"
#include "embedding.h"
#include "gradient.h"
#include "../../util/logger/logger.h"

#include <torch/torch.h>

class Model {
public:
    Model(const int vocabSize, const int seqLen, const int dims, const int layers, const int heads) :
        _embedding(vocabSize, seqLen, dims),
        _finalForm(dims),
        _outputProjection(dims, vocabSize) {
        _blocks.reserve(layers);
        for (int i = 0; i < layers; i++) {
            _blocks.emplace_back(dims, heads);
        }
    }

    ~Model() = default;

public:
    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        auto ep = _embedding.parameters();
        params.insert(params.end(), ep.begin(), ep.end());

        for (auto &block: _blocks) {
            auto bp = block.parameters();
            params.insert(params.end(), bp.begin(), bp.end());
        }

        auto fp = _finalForm.parameters();
        params.insert(params.end(), fp.begin(), fp.end());

        auto op = _outputProjection.parameters();
        params.insert(params.end(), op.begin(), op.end());

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        auto eg = _embedding.gradients();
        grads.insert(grads.end(), eg.begin(), eg.end());

        for (auto &block: _blocks) {
            auto bg = block.gradients();
            grads.insert(grads.end(), bg.begin(), bg.end());
        }

        auto fg = _finalForm.gradients();
        grads.insert(grads.end(), fg.begin(), fg.end());

        auto og = _outputProjection.gradients();
        grads.insert(grads.end(), og.begin(), og.end());

        return grads;
    }

    void zeroGrad() {
        _embedding.zeroGrad();
    }

    torch::Tensor loss(const torch::Tensor &tokenIds) {
        const auto input = tokenIds.slice(0, 0, tokenIds.size(0) - 1);
        const auto targets = tokenIds.slice(0, 1, tokenIds.size(0));

        const auto logits = forward(input);

        const auto [loss, _] = Gradient::crossEntropy(logits, targets);

        return loss;
    }

    torch::Tensor forward(const torch::Tensor &tokenIds) {
        auto x = _embedding.forward(tokenIds);
        for (auto &block: _blocks) {
            x = block.forward(x);
        }
        x = _finalForm.forward(x);

        // update our output projection with the x content window which stores the dimensions for all vocab words
        // [dims, vocabs] = [0] => [each embedding value for each vocab word
        // W = [256, 4000] (output projection)
        // x = [64, 256] - our residual
        // for each of the 64 (seqLen) positions:
        //     for each of the 256 (output projection, inner) dims:
        //         for each of the 4000 vocab words:
        //             logits[position][word] += x[position][dim] * W[dim][word]
        //
        //     logits[position] += bias  // add bias [4000] to each position
        // logits = 1 score per vocab word
        _logits = _outputProjection.forward(x);

        return _logits;
    }

    void backward(const torch::Tensor &tokenIds) {
        auto inputs = tokenIds.slice(0, 0, tokenIds.size(0) - 1);
        const auto targets = tokenIds.slice(0, 1, tokenIds.size(0));

        const auto [_, dLogits] = Gradient::crossEntropy(_logits, targets);

        auto derivativeX = _outputProjection.backward(dLogits);

        derivativeX = _finalForm.backward(derivativeX);

        for (int i = static_cast<int>(_blocks.size()) - 1; i >= 0; i--)
            derivativeX = _blocks[i].backward(derivativeX);

        _embedding.backward(derivativeX);
    }
private:
    torch::Tensor _logits;

    Embedding _embedding;

    LayerNorm _finalForm;

    Linear _outputProjection;
    std::vector<Block> _blocks;
};
