#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

Dataset load_csv(const std::string& filepath) {
    Dataset ds;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir " << filepath << "\n";
        return ds;
    }

    std::string line;
    std::getline(file, line); // saltar encabezado

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        Sample s;

        // inTemp, inHumid, outTemp, outHumid
        std::getline(ss, token, ','); s.inTemp   = std::stod(token);
        std::getline(ss, token, ','); s.inHumid  = std::stod(token);
        std::getline(ss, token, ','); s.outTemp  = std::stod(token);
        std::getline(ss, token, ','); s.outHumid = std::stod(token);

        // door, power — saltar
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');

        // battery
        std::getline(ss, token, ','); s.battery  = std::stod(token);

        ds.samples.push_back(s);
    }

    return ds;
}