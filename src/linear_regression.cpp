#include "linear_regression.h"
#include <iostream>
#include <thread>
#include <vector>

// ── Predict ──────────────────────────────────
double LinearRegression::predict(const Sample& s) const {
    return w0 + w1*s.inHumid + w2*s.outTemp + w3*s.battery;
}

// ── MSE ──────────────────────────────────────
double LinearRegression::mse(const Dataset& ds) const {
    double sum = 0;
    for (const auto& s : ds.samples) {
        double error = predict(s) - s.inTemp;
        sum += error * error;
    }
    return sum / ds.size();
}

// ── Fit Secuencial ───────────────────────────
void LinearRegression::fit(const Dataset& ds, double lr, int epochs) {
    w0 = w1 = w2 = w3 = 0.0;
    int N = ds.size();

    for (int e = 0; e < epochs; e++) {
        double g0=0, g1=0, g2=0, g3=0;

        for (int i = 0; i < N; i++) {
            const Sample& s = ds.samples[i];
            double error = predict(s) - s.inTemp;
            g0 += error;
            g1 += error * s.inHumid;
            g2 += error * s.outTemp;
            g3 += error * s.battery;
        }

        w0 -= lr * (g0 / N);
        w1 -= lr * (g1 / N);
        w2 -= lr * (g2 / N);
        w3 -= lr * (g3 / N);

        // if (e % 50 == 0)
        //     std::cout << "Epoca " << e << "  MSE=" << mse(ds) << "\n";
    }
}

// ── Fit Paralelo ─────────────────────────────
void LinearRegression::fit_parallel(const Dataset& ds, int n_threads,
                                     double lr, int epochs) {
    w0 = w1 = w2 = w3 = 0.0;
    int N = ds.size();
    int chunk = N / n_threads;

    for (int e = 0; e < epochs; e++) {
        std::vector<double> g0(n_threads, 0);
        std::vector<double> g1(n_threads, 0);
        std::vector<double> g2(n_threads, 0);
        std::vector<double> g3(n_threads, 0);

        std::vector<std::thread> hilos;
        for (int t = 0; t < n_threads; t++) {
            hilos.emplace_back([&, t]() {
                int inicio = t * chunk;
                int fin    = (t == n_threads-1) ? N : inicio + chunk;

                for (int i = inicio; i < fin; i++) {
                    const Sample& s = ds.samples[i];
                    double error = predict(s) - s.inTemp;
                    g0[t] += error;
                    g1[t] += error * s.inHumid;
                    g2[t] += error * s.outTemp;
                    g3[t] += error * s.battery;
                }
            });
        }

        for (auto& h : hilos) h.join();

        double tot0=0, tot1=0, tot2=0, tot3=0;
        for (int t = 0; t < n_threads; t++) {
            tot0 += g0[t]; tot1 += g1[t];
            tot2 += g2[t]; tot3 += g3[t];
        }

        w0 -= lr * (tot0 / N);
        w1 -= lr * (tot1 / N);
        w2 -= lr * (tot2 / N);
        w3 -= lr * (tot3 / N);
    }
}