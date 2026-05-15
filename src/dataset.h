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
struct Scaler {
    double inHumid_min, inHumid_max;
    double outTemp_min, outTemp_max;
    double battery_min, battery_max;
    double inTemp_min,  inTemp_max;

    // Calcula min y max de cada columna
    void fit(const Dataset& ds) {
        inHumid_min = inHumid_max = ds.samples[0].inHumid;
        outTemp_min = outTemp_max = ds.samples[0].outTemp;
        battery_min = battery_max = ds.samples[0].battery;
        inTemp_min  = inTemp_max  = ds.samples[0].inTemp;

        for (const auto& s : ds.samples) {
            if (s.inHumid < inHumid_min) inHumid_min = s.inHumid;
            if (s.inHumid > inHumid_max) inHumid_max = s.inHumid;
            if (s.outTemp < outTemp_min) outTemp_min = s.outTemp;
            if (s.outTemp > outTemp_max) outTemp_max = s.outTemp;
            if (s.battery < battery_min) battery_min = s.battery;
            if (s.battery > battery_max) battery_max = s.battery;
            if (s.inTemp  < inTemp_min)  inTemp_min  = s.inTemp;
            if (s.inTemp  > inTemp_max)  inTemp_max  = s.inTemp;
        }
    }

    // Normaliza un sample
    Sample transform(const Sample& s) const {
        Sample n = s;
        n.inHumid = (s.inHumid - inHumid_min) / (inHumid_max - inHumid_min);
        n.outTemp = (s.outTemp - outTemp_min) / (outTemp_max - outTemp_min);
        n.battery = (s.battery - battery_min) / (battery_max - battery_min);
        n.inTemp  = (s.inTemp  - inTemp_min)  / (inTemp_max  - inTemp_min);
        return n;
    }
};

Dataset load_csv(const std::string& filepath);