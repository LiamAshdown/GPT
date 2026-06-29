#pragma once

#include <math.h>
#include <random>

class Math {
public:
    static int random(const int min, const int max) {
        std::random_device random;
        std::mt19937 gen(random());
        std::uniform_int_distribution<int> dist(min, max);

        return dist(gen);
    }
};
