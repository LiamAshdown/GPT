#pragma once

#include "../../util/math.h"
#include "../../util/event-dispatcher/event-dispatcher.h"
#include "../../util/event-dispatcher/listeners/sample-window-listener.h"

#include <vector>
#include <torch/torch.h>
#include "device.h"

class Window {
public:
    Window(const int seqLen, const int batchSize = 2, EventDispatcher* dispatcher = nullptr) :
        _batchSize(batchSize), _seqLen(seqLen), _dispatcher(dispatcher) {
    }
    ~Window() = default;

    // returns a vector of batchSize sequences, each of length seqLen+1
    std::vector<torch::Tensor> sample(const std::vector<int>& tokens) const {
        // We need to subtract seq len from token size so we ensure we are always in bound
        // of the context window
        const int maxEnd = static_cast<int>(tokens.size() - _seqLen);

        std::vector<torch::Tensor> batch;

        for (int i = 0; i < _batchSize; i++) {
            const int start = Math::random(0, maxEnd);
            const int end = start + _seqLen + 1;

            std::vector<int> window(tokens.begin() + start, tokens.begin() + end);

            batch.push_back(torch::tensor(window, torch::TensorOptions().dtype(torch::kLong).device(globalDevice())));

            if (_dispatcher) {
                _dispatcher->emit<SampleWindowEvent>({start, end, window});
            }
        }

        return batch;
    }

    int batchSize() const { return _batchSize; }
    int seqLen() const { return _seqLen; }

private:
    int _batchSize;
    int _seqLen;

    EventDispatcher* _dispatcher;
};