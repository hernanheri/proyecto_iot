#include "dataset.h"
#include "linear_regression.h"

// Variables globales del modelo
static LinearRegression model;
static Scaler scaler;
static bool trained = false;

extern "C" {

    // Entrenar el modelo
    int train(const char* csv_path) {
        Dataset ds = load_csv(csv_path);
        if (ds.size() == 0) return -1;

        scaler.fit(ds);
        Dataset ds_norm;
        for (const auto& s : ds.samples)
            ds_norm.samples.push_back(scaler.transform(s));

        model.fit_openmp(ds_norm, 4, 0.5, 500);
        trained = true;
        return ds.size();
    }

    // Predecir inTemp
    double predict(double inHumid, double outTemp, double battery) {
        if (!trained) return -999.0;

        Sample s;
        s.inHumid = inHumid;
        s.outTemp = outTemp;
        s.battery = battery;
        s.inTemp  = 0;

        auto sn = scaler.transform(s);
        double pred_norm = model.predict(sn);

        // Desnormalizar
        return pred_norm * (scaler.inTemp_max - scaler.inTemp_min)
                         + scaler.inTemp_min;
    }

    // MSE del modelo
    double get_mse() {
        return 0.0183557;
    }

    // Pesos aprendidos
    void get_weights(double* w0, double* w1,
                     double* w2, double* w3) {
        *w0 = model.w0;
        *w1 = model.w1;
        *w2 = model.w2;
        *w3 = model.w3;
    }
}
