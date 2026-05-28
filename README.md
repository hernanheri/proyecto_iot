# Sistema IoT — Análisis y Predicción con IA Paralela

Proyecto de análisis y predicción de datos de sensores IoT usando
regresión lineal implementada de forma secuencial, paralela y distribuida.

---

## Descripción

Se analizan datos reales de sensores de refrigerador (temperatura interior,
humedad, temperatura exterior, batería) para predecir la temperatura
interior usando gradiente descendiente. Se compara el desempeño entre
versión secuencial, paralela (std::thread y OpenMP) y distribuida (MPI).
El modelo se expone como librería y se consume desde un frontend Python.

---

## Estructura del Proyecto
proyecto_iot/
├── src/
│   ├── main.cpp                    # Experimentos seq + paralelo
│   ├── linear_regression_mpi.cpp   # Versión distribuida MPI
│   ├── iot_lib.cpp                 # Librería .so para Python
│   ├── dataset.h / dataset.cpp     # Lectura y normalización
│   ├── linear_regression.h/.cpp    # Algoritmo (seq + threads + OpenMP)
│   ├── timer.h                     # Medición de tiempos
│   └── Makefile
├── data/
│   └── 18DAYSDATA_corrected.csv    # 555,804 muestras reales
├── results/
│   ├── resultados.csv              # Tiempos seq/threads/openmp
│   ├── resultados_mpi.csv          # Tiempos MPI
│   └── grafica_desempeno.png
├── app.py                          # Frontend Python (tkinter)
├── visualize.py                    # Gráficas de desempeño
└── README.md

---

## Dataset

Datos reales de sensores IoT con 555,804 muestras.

| Variable  | Descripción           | Rango         |
|-----------|-----------------------|---------------|
| inTemp    | Temperatura interior  | -3 a 6.3°C    |
| inHumid   | Humedad interior      | 85 a 95%      |
| outTemp   | Temperatura exterior  | 24.4 a 37.6°C |
| battery   | Nivel de batería      | 3.44 a 4.5V   |

**Objetivo:** predecir `inTemp` a partir de las demás variables.

---

## Algoritmo — Regresión Lineal

Implementación desde cero con gradiente descendiente batch:
y_pred = w0 + w1inHumid + w2outTemp + w3*battery
error  = y_pred - y_real
w_i    = w_i - lr * mean(grad_i)

Parámetros: `lr = 0.5`, `epochs = 500`, `MSE final = 0.0183`

---

## Paralelización

### std::thread
- División manual del dataset en chunks por hilo
- Cada hilo acumula gradientes locales
- Reducción en el hilo principal al final de cada época

### OpenMP
- `#pragma omp parallel for reduction`
- Pool de hilos reutilizado entre épocas
- Planificación estática

### MPI
- Distribución del dataset entre procesos
- Cada proceso calcula gradientes locales
- `MPI_Allreduce` sincroniza gradientes entre procesos
- Todos los procesos actualizan los mismos pesos

---

## Resultados

### Secuencial vs Paralelo (CPU local)

| Versión        | Tiempo (ms) | Speedup | Eficiencia |
|----------------|-------------|---------|------------|
| Secuencial     | 712         | 1.00x   | 1.00       |
| std::thread 2h | 6177        | 0.11x   | 0.06       |
| std::thread 4h | 11321       | 0.06x   | 0.02       |
| OpenMP 2h      | 497         | 1.43x   | 0.71       |
| OpenMP 4h      | 457         | 1.55x   | 0.39       |

### MPI Distribuido

| Procesos | Tiempo (ms) | Speedup |
|----------|-------------|---------|
| 1        | 518         | 1.00x   |
| 2        | 420         | 1.23x   |
| 4        | 430         | 1.20x   |

### Conclusiones
- OpenMP supera al secuencial gracias a su pool de hilos interno
- std::thread es más lento por overhead de creación de hilos por época
- MPI escala bien a 2 procesos; la comunicación limita la ganancia a 4
- Resultados consistentes con la Ley de Amdahl

---

## Compilación y Ejecución

### Requisitos
- g++ con soporte C++17 y OpenMP
- OpenMPI
- Python 3 con matplotlib, pandas y tkinter

### Compilar todo
```bash
cd src
make          # versión secuencial + paralela
make mpi      # versión MPI
make lib      # librería .so para Python
```

### Ejecutar experimentos
```bash
./main
```

### Ejecutar versión MPI
```bash
mpirun -np 1 ./main_mpi
mpirun -np 2 ./main_mpi
mpirun -np 4 ./main_mpi
```

### Generar gráficas
```bash
cd ..
python3 visualize.py
```

### Ejecutar frontend
```bash
python3 app.py
```

---

## Tecnologías

- C++17
- OpenMP
- std::thread
- MPI (OpenMPI)
- Python 3
- matplotlib / pandas / tkinter