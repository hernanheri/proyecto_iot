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

    // Normalizar
    Scaler sc;
    sc.fit(ds);
    std::cout << "Rangos encontrados:\n";
    std::cout << "  inTemp:  " << sc.inTemp_min  << " a " << sc.inTemp_max  << "\n";
    std::cout << "  inHumid: " << sc.inHumid_min << " a " << sc.inHumid_max << "\n";
    std::cout << "  outTemp: " << sc.outTemp_min << " a " << sc.outTemp_max << "\n";
    std::cout << "  battery: " << sc.battery_min << " a " << sc.battery_max << "\n";
    Dataset ds_norm;
    for (const auto& s : ds.samples)
        ds_norm.samples.push_back(sc.transform(s));

    // Entrenar con datos normalizados
    LinearRegression model;
    model.fit(ds_norm, 0.5, 500);  // lr más grande, más épocas

    // Prueba con las primeras 3 muestras
    // Desnormalizar predicción para ver en °C reales
    std::cout << "\nPredicciones vs valores reales:\n";
    for (int i = 0; i < 5; i++) {
        double pred_norm = model.predict(ds_norm.samples[i]);
        // Desnormalizar: volver a escala original
        double pred_real = pred_norm * (sc.inTemp_max - sc.inTemp_min)
                           + sc.inTemp_min;
        std::cout << "  Real: " << ds.samples[i].inTemp
                  << "°C  Predicho: " << pred_real << "°C\n";
    }

    return 0;
}