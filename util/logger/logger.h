#pragma once

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <torch/torch.h>

inline std::string tensorStr(const torch::Tensor& t) {
    std::ostringstream ss;
    ss << t;
    return ss.str();
}