#pragma once

#include "../events/events.h"
#include "../../logger/logger.h"



class GptListener {
public:
    GptListener(const std::shared_ptr<spdlog::logger> &logger): _logger(logger) {}

    void onBatchLoss(const BatchLossEvent& event) {
        _accLoss += event.batchLoss;
        _accCount++;
    }

    void onStep(const StepEvent& event) {
        if (event.step % 100 == 0) {
            const float avg = _accCount > 0 ? _accLoss / _accCount : 0.0f;
            _logger->info("Step: {:5d} / {}  |  avg loss: {:.4f}", event.step, event.steps, avg);
            _accLoss  = 0.0f;
            _accCount = 0;
        }
    }

private:
    std::shared_ptr<spdlog::logger> _logger;
    float _accLoss  = 0.0f;
    int   _accCount = 0;
};