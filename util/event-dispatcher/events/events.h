#pragma once

#include "../event-dispatcher.h"

#include <vector>

struct SampleWindowEvent : Event {
    SampleWindowEvent(const int start, const int end, const std::vector<int> tokens) :
        start(start), end(end), tokens(tokens) {}

    int start;
    int end;

    std::vector<int> tokens;
};

struct BatchLossEvent : Event {
    BatchLossEvent(const float batchLoss) : batchLoss(batchLoss) {}

    int batchLoss;
};

struct StepEvent : Event {
    StepEvent(const int step, const int steps) : step(step), steps(steps) {}

    int step;
    int steps;
};