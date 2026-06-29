#pragma once

#include <torch/torch.h>
#include "device.h"

static constexpr float WEIGHT_INIT_SCALE = 0.02f;

class Embedding {
public:
    Embedding(const int vocabSize, const int maxSeqLen, const int dims) : _dims(dims) {
        // Token embedding lookup table
        _tokenW = torch::randn({ vocabSize, dims }, globalDevice()) * WEIGHT_INIT_SCALE;
        // Token position look up table
        _posW = torch::randn({ maxSeqLen, dims }, globalDevice()) * WEIGHT_INIT_SCALE;

        _derivativeTokenW = torch::zeros({ vocabSize, dims }, globalDevice());
        _derivativePosW   = torch::zeros({ maxSeqLen, dims }, globalDevice());
    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        params.push_back(_tokenW);
        params.push_back(_posW);

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        grads.push_back(_derivativeTokenW);
        grads.push_back(_derivativePosW);

        return grads;
    }

    torch::Tensor forward(const torch::Tensor& tokenIds) {
        _tokenIds = tokenIds;

        // create an array from token size, e.g 128 -> [0, 1, 2, 3, 4, ... 127]
        auto positions = torch::arange(tokenIds.size(0), torch::TensorOptions().dtype(torch::kLong).device(globalDevice()));

        // from the list of token ids, we need to turn them into an array where each token
        // holds the dimensions, e.g _tokenW[tokenId] = [dimensions]
        // we also add position embeddings so the model knows where each token is in the sequence.
        // the same token at different positions gets a different vector:
        //
        // "the cat sat on the mat"
        //   |              |
        //  "the" at pos 0   "the" at pos 4  <- same token, different positions
        //
        // tokenW["the"] alone would give identical vectors for both.
        // adding posW[0] and posW[4] makes them unique so the model
        // understands word order - "cat sat" means something different to "sat cat"
        return _tokenW.index({tokenIds}) + _posW.index({positions});
    }

    /*
     * Comment below is summarised by AI to help me understand:
     *
     * index_add_ adds gradients back to the correct token rows.
     *
     * Say vocab has 5 tokens, dims=3:
     *
     * _derivativeTokenW = [[0, 0, 0],   <- token 0
     *                      [0, 0, 0],   <- token 1
     *                      [0, 0, 0],   <- token 2
     *                      [0, 0, 0],   <- token 3
     *                      [0, 0, 0]]   <- token 4
     *
     * Your window had these tokens:
     * tokenIds = [2, 4, 2]  <- token 2, token 4, token 2
     *
     * Gradients coming back:
     * dOut = [[0.1, 0.2, 0.3],  <- gradient for position 0 (token 2)
     *         [0.4, 0.5, 0.6],  <- gradient for position 1 (token 4)
     *         [0.7, 0.8, 0.9]]  <- gradient for position 2 (token 2 again)
     *
     * index_add_ adds each gradient to the correct token row:
     * token 2 gets dOut[0]: [0,   0,   0  ] += [0.1, 0.2, 0.3] = [0.1, 0.2, 0.3]
     * token 4 gets dOut[1]: [0,   0,   0  ] += [0.4, 0.5, 0.6] = [0.4, 0.5, 0.6]
     * token 2 gets dOut[2]: [0.1, 0.2, 0.3] += [0.7, 0.8, 0.9] = [0.8, 1.0, 1.2]
     *
     * Result:
     * _derivativeTokenW = [[0,   0,   0  ],  <- token 0 (not in window, unchanged)
     *                      [0,   0,   0  ],  <- token 1 (not in window, unchanged)
     *                      [0.8, 1.0, 1.2],  <- token 2 (appeared twice, accumulated)
     *                      [0,   0,   0  ],  <- token 3 (not in window, unchanged)
     *                      [0.4, 0.5, 0.6]]  <- token 4 (appeared once)
     *
     * Only tokens that appeared in the window get gradients — because
     * only those tokens were used in the forward pass.
     */
    torch::Tensor backward(torch::Tensor dOut) {
        const auto positions = torch::arange(_tokenIds.size(0), torch::TensorOptions().dtype(torch::kLong).device(globalDevice()));

        _derivativeTokenW.index_add_(0, _tokenIds, dOut);
        _derivativePosW.index_add_(0, positions, dOut);
        return dOut;
    }

    void zeroGrad() {
        _derivativeTokenW.zero_();
        _derivativePosW.zero_();
    }
private:
    int _dims;

    torch::Tensor _tokenW;
    torch::Tensor _posW;

    torch::Tensor _tokenIds;
    torch::Tensor _derivativeTokenW;
    torch::Tensor _derivativePosW;
};