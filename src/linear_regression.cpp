#include "linear_regression.h"
#include <iostream>
#include <cmath>

double LinearRegression::predict(const Sample& s) const {
    return w0 + w1*s.inHumid + w2*s.outTemp + w3*s.battery;
}

void LinearRegression::fit(const Dataset& ds, double lr, int epochs) {
    int N = ds.size();

    for (int e = 0; e < epochs; e++) {
        double grad0 = 0, grad1 = 0, grad2 = 0, grad3 = 0;

        // Acumular gradientes sobre todas las muestras
        for (int i = 0; i < N; i++) {
            const Sample& s = ds.samples[i];
            double error = predict(s) - s.inTemp;

            grad0 += error;
            grad1 += error * s.inHumid;
            grad2 += error * s.outTemp;
            grad3 += error * s.battery;
        }

        // Actualizar pesos con el gradiente promedio
        w0 -= lr * grad0 / N;
        w1 -= lr * grad1 / N;
        w2 -= lr * grad2 / N;
        w3 -= lr * grad3 / N;

        // Mostrar progreso cada 5 épocas
        if (e % 50 == 0) {
            std::cout << "Epoca " << e
                    << "  MSE=" << mse(ds) << "\n";
        }
    }
}

double LinearRegression::mse(const Dataset& ds) const {
    double total = 0;
    for (const auto& s : ds.samples) {
        double error = predict(s) - s.inTemp;
        total += error * error;
    }
    return total / ds.size();
}