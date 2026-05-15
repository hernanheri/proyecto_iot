#pragma once
#include "dataset.h"

struct LinearRegression {
    double w0 = 0;  // bias
    double w1 = 0;  // peso inHumid
    double w2 = 0;  // peso outTemp
    double w3 = 0;  // peso battery

    // Entrena el modelo (versión secuencial)
    void fit(const Dataset& ds, double lr = 0.0001, int epochs = 20);

    // Predice inTemp dado una muestra
    double predict(const Sample& s) const;

    // Calcula el MSE sobre todo el dataset
    double mse(const Dataset& ds) const;

    void fit_parallel(const Dataset& ds, int n_threads,
                  double lr = 0.5, int epochs = 500);
    
    void fit_openmp(const Dataset& ds, int n_threads,
                    double lr = 0.5, int epochs = 500);
};