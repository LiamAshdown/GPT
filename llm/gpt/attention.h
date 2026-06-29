#pragma once
#include "linear.h"
#include "device.h"
#include <limits>

class Attention {
public:
    Attention(const int heads, const int dims) :
        _heads(heads),
        _dims(dims),
        _dimsByHead(dims / heads),
        _W_I(dims, dims * 3),
        _W_O(dims, dims) {
    }

    std::vector<std::reference_wrapper<torch::Tensor>> parameters() {
        std::vector<std::reference_wrapper<torch::Tensor>> params;

        auto wi = _W_I.parameters();
        params.insert(params.end(), wi.begin(), wi.end());

        auto wo = _W_O.parameters();
        params.insert(params.end(), wo.begin(), wo.end());

        return params;
    }

    std::vector<std::reference_wrapper<torch::Tensor>> gradients() {
        std::vector<std::reference_wrapper<torch::Tensor>> grads;

        auto wi = _W_I.gradients();
        grads.insert(grads.end(), wi.begin(), wi.end());

        auto wo = _W_O.gradients();
        grads.insert(grads.end(), wo.begin(), wo.end());

        return grads;
    }

    torch::Tensor forward(torch::Tensor x) {
        _x = x;
        const int seqLen = x.size(0);

        const auto qkv = _W_I.forward(x).chunk(3, 1);

        // https://poloclub.github.io/transformer-explainer/
        // In my own lingo:
        // Take for example this sentence: "the cat sat"
        // Q ("what am I looking for?")
        //   - "sat" is a verb, its Q vector asks "who or what is doing this action?"
        //
        // K ("what do I contain?")
        //   - "cat" broadcasts "I am a noun/subject"
        //   - "the" broadcasts "I am an article"
        //
        // V ("my actual content")
        //   - the real information each token carries, pulled in proportion to Q·K scores
        //
        // Score = Q["sat"] · K["cat"] → high  (cat is relevant to sat)
        // Score = Q["sat"] · K["the"] → low   (the is not relevant to sat)
        //
        // softmax(scores) -> weights [0.05, 0.78, 0.17]  (the, cat, sat)
        //
        // output["sat"] = 0.05*V["the"] + 0.78*V["cat"] + 0.17*V["sat"]
        //               -> "sat" now knows it is acting on "cat"
        _Q = qkv[0].reshape({_heads, seqLen, _dimsByHead});
        _K = qkv[1].reshape({_heads, seqLen, _dimsByHead});
        _V = qkv[2].reshape({_heads, seqLen, _dimsByHead});

        // divide by sqrt(dimsByHead) to prevent scores growing too large as dims increase
        // large scores -> softmax saturates-> gradients vanish -> model stops learning
        auto scores = torch::matmul(_Q, _K.transpose(1, 2)) / std::sqrt(static_cast<float>(_dimsByHead));

        // Mask the scores so the attention doesn't calculate the next token
        // otherwise this would be cheating and the model wouldn't be learning
        //         the  cat  sat  on
        // the    [  0    *   *   * ]
        // cat    [  0    0   *   * ]
        // sat    [  0    0   0   * ]
        // on     [  0    0   0   0 ]
        // 0 = random score
        const auto mask = torch::triu(torch::full({seqLen, seqLen}, -std::numeric_limits<float>::infinity(), globalDevice()), 1);

        scores = scores + mask;

        // turn it into probabilities
        _weights = torch::softmax(scores, 2);

        const auto out = torch::matmul(_weights, _V);

        // [4, 128, 32] -> [128, 128]
        // concat QKV
        const auto concat = out.transpose(0, 1).contiguous().view({seqLen, _dims});

        return _W_O.forward(concat);
    }

    torch::Tensor backward(const torch::Tensor &dOut) {
        int seqLen = _x.size(0);

        auto dConcat = _W_O.backward(dOut);
        auto dHeadOut = dConcat.view({seqLen, _heads, _dimsByHead}).transpose(0, 1);

        auto dV = torch::matmul(_weights.transpose(1, 2), dHeadOut);
        auto dWeights = torch::matmul(dHeadOut, _V.transpose(1, 2));

        auto dScores = _weights * (dWeights - (dWeights * _weights).sum(-1, true));
        dScores = dScores / std::sqrt(static_cast<float>(_dimsByHead));

        auto dQ = torch::matmul(dScores, _K);
        auto dK = torch::matmul(dScores.transpose(1, 2), _Q);

        auto dQKV = torch::cat({
                                   dQ.transpose(0, 1).contiguous().view({seqLen, _dims}),
                                   dK.transpose(0, 1).contiguous().view({seqLen, _dims}),
                                   dV.transpose(0, 1).contiguous().view({seqLen, _dims})
                               }, 1);

        return _W_I.backward(dQKV);
    }

private:
    int _heads;
    int _dims;
    int _dimsByHead;

    torch::Tensor _x;

    torch::Tensor _Q;
    torch::Tensor _K;
    torch::Tensor _V;

    torch::Tensor _weights;

    // Input Projection
    Linear _W_I;
    // Output Projection
    Linear _W_O;
};
