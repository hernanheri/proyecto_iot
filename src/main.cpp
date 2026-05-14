#include <iostream>
#include "dataset.h"
#include "linear_regression.h"

int main() {
    Dataset ds = load_csv("../data/18DAYSDATA_corrected.csv");

    if (ds.size() == 0) {
        std::cerr << "Dataset vacío.\n";
        return 1;
    }

    std::cout << "Muestras cargadas: " << ds.size() << "\n\n";

    // Entrenar
    LinearRegression model;
    model.fit(ds, 0.0001, 20);

    // Ver pesos finales
    std::cout << "\nPesos finales:\n";
    std::cout << "  w0 (bias)    = " << model.w0 << "\n";
    std::cout << "  w1 (inHumid) = " << model.w1 << "\n";
    std::cout << "  w2 (outTemp) = " << model.w2 << "\n";
    std::cout << "  w3 (battery) = " << model.w3 << "\n";

    // Probar con la primera muestra
    std::cout << "\nPrueba con muestra 0:\n";
    std::cout << "  Real:     " << ds.samples[0].inTemp << "\n";
    std::cout << "  Predicho: " << model.predict(ds.samples[0]) << "\n";

    return 0;
}