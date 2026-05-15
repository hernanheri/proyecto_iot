import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/resultados.csv")
print(df)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Regresión Lineal — Análisis de Desempeño Paralelo", fontsize=13)

# ── Gráfica 1: Tiempo ──
ax1 = axes[0]
colores = {'secuencial': 'steelblue', 'threads': 'tomato', 'openmp': 'seagreen'}
for version in df['version'].unique():
    sub = df[df['version'] == version]
    ax1.bar([f"{version}\n{int(h)}h" for h in sub['n_hilos']],
            sub['tiempo_ms'], color=colores[version], alpha=0.85)
ax1.set_title("Tiempo de Ejecución")
ax1.set_ylabel("Tiempo (ms)")
ax1.set_xlabel("Versión")

# ── Gráfica 2: Speedup ──
ax2 = axes[1]
omp = df[df['version'] == 'openmp']
thr = df[df['version'] == 'threads']
hilos = [2, 4]
ax2.plot(hilos, omp['speedup'].values, 'o-', color='seagreen', label='OpenMP', lw=2)
ax2.plot(hilos, thr['speedup'].values, 's-', color='tomato', label='std::thread', lw=2)
ax2.plot(hilos, hilos, 'k--', alpha=0.4, label='Ideal')
ax2.set_title("Speedup vs Nº Hilos")
ax2.set_xlabel("Número de hilos")
ax2.set_ylabel("Speedup")
ax2.legend()
ax2.set_xticks(hilos)

# ── Gráfica 3: Eficiencia ──
ax3 = axes[2]
ax3.plot(hilos, omp['eficiencia'].values, 'o-', color='seagreen', label='OpenMP', lw=2)
ax3.plot(hilos, thr['eficiencia'].values, 's-', color='tomato', label='std::thread', lw=2)
ax3.axhline(1.0, color='k', ls='--', alpha=0.4, label='Ideal')
ax3.set_title("Eficiencia Paralela")
ax3.set_xlabel("Número de hilos")
ax3.set_ylabel("Eficiencia")
ax3.legend()
ax3.set_xticks(hilos)
ax3.set_ylim(0, 1.3)

plt.tight_layout()
plt.savefig("results/grafica_desempeno.png", dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada en results/grafica_desempeno.png")