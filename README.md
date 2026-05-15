# Sistema IoT — Análisis y Predicción con IA Paralela

Proyecto de análisis y predicción de datos de sensores IoT usando
regresión lineal implementada de forma secuencial y paralela en C++.

---

## Descripción

Se analizan datos reales de sensores ambientales (temperatura interior,
humedad, temperatura exterior, batería) para predecir la temperatura
interior usando gradiente descendiente. Se compara el desempeño entre
la versión secuencial y dos estrategias de paralelización: std::thread
y OpenMP.

---

## Estructura del Proyecto
proyecto_iot/
├── src/
│   ├── main.cpp                # Punto de entrada y experimentos
│   ├── dataset.h / dataset.cpp # Lectura y normalización del CSV
│   ├── linear_regression.h     # Regresión lineal (seq + paralela)
│   ├── linear_regression.cpp
│   ├── timer.h                 # Medición de tiempos
│   └── Makefile
├── data/
│   └── 18DAYSDATA_corrected.csv
├── results/
│   ├── resultados.csv
│   └── grafica_desempeno.png
├── visualize.py
└── README.md

---

## Dataset

Datos reales de sensores IoT con 555,804 muestras y 10 columnas.
Variables utilizadas:

| Variable  | Descripción          | Rango       |
|-----------|----------------------|-------------|
| inTemp    | Temperatura interior | -3 a 6.3°C  |
| inHumid   | Humedad interior     | 85 a 95%    |
| outTemp   | Temperatura exterior | 24.4 a 37.6°C |
| battery   | Nivel de batería     | 3.44 a 4.5V |

**Objetivo:** predecir `inTemp` a partir de las demás variables.

---

## Algoritmo — Regresión Lineal

Implementación desde cero usando gradiente descendiente batch:
y_pred = w0 + w1inHumid + w2outTemp + w3*battery
error  = y_pred - y_real
w_i    = w_i - lr * mean(grad_i)

Parámetros: `lr = 0.5`, `epochs = 500`

---

## Paralelización

### std::thread
- División manual del dataset en chunks por hilo
- Cada hilo acumula gradientes locales
- Reducción en el hilo principal al final de cada época

### OpenMP
- Directiva `#pragma omp parallel for reduction`
- Planificación estática
- Pool de hilos reutilizado entre épocas

---

## Resultados

| Versión       | Tiempo (ms) | Speedup | Eficiencia |
|---------------|-------------|---------|------------|
| Secuencial    | 712         | 1.00x   | 1.00       |
| std::thread 2h| 6177        | 0.11x   | 0.06       |
| std::thread 4h| 11321       | 0.06x   | 0.02       |
| OpenMP 2h     | 497         | 1.43x   | 0.71       |
| OpenMP 4h     | 457         | 1.55x   | 0.39       |

**MSE final:** 0.0183 en todas las versiones (resultado idéntico).

### Conclusiones
- OpenMP supera al secuencial gracias a su pool de hilos interno
- std::thread es más lento por el overhead de crear hilos en cada época
- La eficiencia baja de 2 a 4 hilos por la fracción serial del algoritmo
- Resultado consistente con la Ley de Amdahl

---

## Compilación y Ejecución

### Requisitos
- g++ con soporte C++17 y OpenMP
- Python 3 con matplotlib y pandas

### Compilar
```bash
cd src
make
```

### Ejecutar
```bash
./main
```

### Generar gráficas
```bash
cd ..
python3 visualize.py
```

---

## Tecnologías
- C++17
- OpenMP
- std::thread
- Python 3 / matplotlib / pandas