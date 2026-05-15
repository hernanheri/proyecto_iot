#pragma once
#include <chrono>

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start;

    void begin() {
        start = Clock::now();
    }

    double elapsed_ms() {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};