#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "dataset.h"
#include "linear_regression.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // ID de este proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total de procesos

    // Solo el proceso 0 carga y normaliza los datos
    Dataset ds, ds_norm;
    Scaler sc;

    if (rank == 0) {
        ds = load_csv("../data/18DAYSDATA_corrected.csv");
        sc.fit(ds);
        for (const auto& s : ds.samples)
            ds_norm.samples.push_back(sc.transform(s));
        std::cout << "Muestras: " << ds_norm.size()
                  << "  Procesos: " << size << "\n";
    }

    // Distribuir tamaño del dataset a todos los procesos
    int N = ds_norm.size();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Dividir datos en chunks por proceso
    int chunk = N / size;
    int inicio = rank * chunk;
    int fin    = (rank == size-1) ? N : inicio + chunk;

    // Enviar datos a cada proceso
    // Aplanar samples en vectores para MPI
    std::vector<double> inTemp_v(N), inHumid_v(N),
                        outTemp_v(N), battery_v(N);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            inTemp_v[i]  = ds_norm.samples[i].inTemp;
            inHumid_v[i] = ds_norm.samples[i].inHumid;
            outTemp_v[i] = ds_norm.samples[i].outTemp;
            battery_v[i] = ds_norm.samples[i].battery;
        }
    }

    // Broadcast de todos los datos
    MPI_Bcast(inTemp_v.data(),  N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(inHumid_v.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(outTemp_v.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(battery_v.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Parámetros
    double lr = 0.5;
    int epochs = 500;
    double w0=0, w1=0, w2=0, w3=0;

    // Medir tiempo en proceso 0
    double t_inicio = MPI_Wtime();

    for (int e = 0; e < epochs; e++) {
        // Gradientes locales de este proceso
        double lg0=0, lg1=0, lg2=0, lg3=0;

        for (int i = inicio; i < fin; i++) {
            double pred  = w0 + w1*inHumid_v[i]
                              + w2*outTemp_v[i]
                              + w3*battery_v[i];
            double error = pred - inTemp_v[i];
            lg0 += error;
            lg1 += error * inHumid_v[i];
            lg2 += error * outTemp_v[i];
            lg3 += error * battery_v[i];
        }

        // Reducción — sumar gradientes de todos los procesos
        double g0=0, g1=0, g2=0, g3=0;
        MPI_Allreduce(&lg0, &g0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&lg1, &g1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&lg2, &g2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&lg3, &g3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Todos los procesos actualizan los mismos pesos
        w0 -= lr * (g0 / N);
        w1 -= lr * (g1 / N);
        w2 -= lr * (g2 / N);
        w3 -= lr * (g3 / N);
    }

    double t_fin = MPI_Wtime();

    if (rank == 0) {
        double tiempo_mpi = (t_fin - t_inicio) * 1000;

        // Calcular MSE final
        double mse = 0;
        for (int i = 0; i < N; i++) {
            double pred  = w0 + w1*inHumid_v[i]
                              + w2*outTemp_v[i]
                              + w3*battery_v[i];
            double err   = pred - inTemp_v[i];
            mse += err * err;
        }
        mse /= N;

        std::cout << "\n--- Resultados MPI ---\n";
        std::cout << "Procesos:  " << size     << "\n";
        std::cout << "Tiempo:    " << tiempo_mpi << " ms\n";
        std::cout << "MSE final: " << mse      << "\n";

        // Guardar en CSV
        std::ofstream csv("../results/resultados_mpi.csv",
                          std::ios::app);
        if (rank == 0) {
            csv << "mpi," << size << "," << tiempo_mpi
                << "," << mse << "\n";
        }
        csv.close();
    }

    MPI_Finalize();
    return 0;
}
