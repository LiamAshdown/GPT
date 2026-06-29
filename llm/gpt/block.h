#pragma once

#include "layer-norm.h"
#include "attention.h"
#include "feed-forward.h"

class Model;

class Block {
    friend class Model;

public:
    Block(const int dims, const int heads) :
        _inputNorm(dims),
        _outputNorm(dims),
        _attention(heads, dims),
        _feedForward(dims) {

    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        auto inp = _inputNorm.parameters();
        params.insert(params.end(), inp.begin(), inp.end());

        auto ap = _attention.parameters();
        params.insert(params.end(), ap.begin(), ap.end());

        auto onp = _outputNorm.parameters();
        params.insert(params.end(), onp.begin(), onp.end());

        auto ffp = _feedForward.parameters();
        params.insert(params.end(), ffp.begin(), ffp.end());

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        auto ing = _inputNorm.gradients();
        grads.insert(grads.end(), ing.begin(), ing.end());

        auto ag = _attention.gradients();
        grads.insert(grads.end(), ag.begin(), ag.end());

        auto ong = _outputNorm.gradients();
        grads.insert(grads.end(), ong.begin(), ong.end());

        auto ffg = _feedForward.gradients();
        grads.insert(grads.end(), ffg.begin(), ffg.end());

        return grads;
    }

    // x = [tokenId] = W * P from embedding
    torch::Tensor forward(const torch::Tensor &x) {
        // "Attention is all you need" was doing post normalization on the residual (x)
        // however after reading other source codes and articles
        // the new modern method is pre and post normalizing.
        const auto xNorm = _inputNorm.forward(x);
        const auto xAttn = x + _attention.forward(xNorm);

        return xAttn + _feedForward.forward(_outputNorm.forward(xAttn));
    }

    torch::Tensor backward(const torch::Tensor& dOut) {
        const auto dAttn2 = dOut;
        const auto dLn2 = _feedForward.backward(dOut);
        const auto dXAttn = dAttn2 + _outputNorm.backward(dLn2);

        const auto dAttn1 = dXAttn;
        const auto dLn1 = _attention.backward(dXAttn);
        return dAttn1 + _inputNorm.backward(dLn1);
    }

private:
    LayerNorm _inputNorm;
    LayerNorm _outputNorm;

    Attention _attention;
    FeedForward _feedForward;
};