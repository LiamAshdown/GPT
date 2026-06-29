#pragma once

#include <torch/torch.h>

inline torch::Device& globalDevice() {
    static torch::Device device(torch::kCPU);
    return device;
}
