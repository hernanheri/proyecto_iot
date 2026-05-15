#include <iostream>
#include "dataset.h"
#include "linear_regression.h"
#include "timer.h"
#include <fstream>

int main() {
    Dataset ds = load_csv("../data/18DAYSDATA_corrected.csv");

    if (ds.size() == 0) {
        std::cerr << "Dataset vacío.\n";
        return 1;
    }

    std::cout << "Muestras: " << ds.size() << "\n";

    // Normalizar
    Scaler sc;
    sc.fit(ds);
    Dataset ds_norm;
    for (const auto& s : ds.samples)
        ds_norm.samples.push_back(sc.transform(s));

    // ── Secuencial ──
    Timer t;
    t.begin();
    LinearRegression model_seq;
    model_seq.fit(ds_norm, 0.5, 500);
    double tiempo_seq = t.elapsed_ms();

    // ── Paralelo 2 hilos ──
    Timer t2;
    t2.begin();
    LinearRegression model_par2;
    model_par2.fit_parallel(ds_norm, 2, 0.5, 500);
    double tiempo_par2 = t2.elapsed_ms();

    // ── Paralelo 4 hilos ──
    Timer t4;
    t4.begin();
    LinearRegression model_par4;
    model_par4.fit_parallel(ds_norm, 4, 0.5, 500);
    double tiempo_par4 = t4.elapsed_ms();

    // ── Resultados ──
    std::cout << "\n--- Tiempos ---\n";
    std::cout << "Secuencial: " << tiempo_seq  << " ms\n";
    std::cout << "2 hilos:    " << tiempo_par2 << " ms\n";
    std::cout << "4 hilos:    " << tiempo_par4 << " ms\n";

    std::cout << "\n--- Speedup ---\n";
    std::cout << "Speedup 2h: " << tiempo_seq/tiempo_par2 << "x\n";
    std::cout << "Speedup 4h: " << tiempo_seq/tiempo_par4 << "x\n";

    std::cout << "\n--- Eficiencia ---\n";
    std::cout << "Eficiencia 2h: " << (tiempo_seq/tiempo_par2)/2 << "\n";
    std::cout << "Eficiencia 4h: " << (tiempo_seq/tiempo_par4)/4 << "\n";

    std::cout << "\n--- MSE final ---\n";
    std::cout << "Secuencial: " << model_seq.mse(ds_norm)  << "\n";
    std::cout << "2 hilos:    " << model_par2.mse(ds_norm) << "\n";
    std::cout << "4 hilos:    " << model_par4.mse(ds_norm) << "\n";

// ── OpenMP 2 hilos ──
    Timer tomp2;
    tomp2.begin();
    LinearRegression model_omp2;
    model_omp2.fit_openmp(ds_norm, 2, 0.5, 500);
    double tiempo_omp2 = tomp2.elapsed_ms();

    // ── OpenMP 4 hilos ──
    Timer tomp4;
    tomp4.begin();
    LinearRegression model_omp4;
    model_omp4.fit_openmp(ds_norm, 4, 0.5, 500);
    double tiempo_omp4 = tomp4.elapsed_ms();

    std::cout << "\n--- OpenMP ---\n";
    std::cout << "2 hilos: " << tiempo_omp2 << " ms  speedup="
              << tiempo_seq/tiempo_omp2 << "x\n";
    std::cout << "4 hilos: " << tiempo_omp4 << " ms  speedup="
              << tiempo_seq/tiempo_omp4 << "x\n";

    std::cout << "\n--- Eficiencia OpenMP ---\n";
    std::cout << "Eficiencia 2h: " << (tiempo_seq/tiempo_omp2)/2 << "\n";
    std::cout << "Eficiencia 4h: " << (tiempo_seq/tiempo_omp4)/4 << "\n";

    std::cout << "\n--- MSE OpenMP ---\n";
    std::cout << "2 hilos: " << model_omp2.mse(ds_norm) << "\n";
    std::cout << "4 hilos: " << model_omp4.mse(ds_norm) << "\n";

    std::ofstream csv("../results/resultados.csv");
    csv << "version,n_hilos,tiempo_ms,speedup,eficiencia\n";
    csv << "secuencial,1,"  << tiempo_seq  << ",1.0,1.0\n";
    csv << "threads,2,"     << tiempo_par2 << "," << tiempo_seq/tiempo_par2 << "," << (tiempo_seq/tiempo_par2)/2 << "\n";
    csv << "threads,4,"     << tiempo_par4 << "," << tiempo_seq/tiempo_par4 << "," << (tiempo_seq/tiempo_par4)/4 << "\n";
    csv << "openmp,2,"      << tiempo_omp2 << "," << tiempo_seq/tiempo_omp2 << "," << (tiempo_seq/tiempo_omp2)/2 << "\n";
    csv << "openmp,4,"      << tiempo_omp4 << "," << tiempo_seq/tiempo_omp4 << "," << (tiempo_seq/tiempo_omp4)/4 << "\n";
    csv.close();
    std::cout << "\nResultados guardados en results/resultados.csv\n";

    return 0;
}