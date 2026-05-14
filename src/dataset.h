#pragma once
#include <vector>
#include <string>

struct Sample {
    double inTemp;
    double inHumid;
    double outTemp;
    double outHumid;
    double battery;
};

struct Dataset {
    std::vector<Sample> samples;
    size_t size() const { return samples.size(); }
};

Dataset load_csv(const std::string& filepath);