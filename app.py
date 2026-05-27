import ctypes
import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading

# ── Cargar librería ──────────────────────────────────────────────
lib_path = os.path.join(os.path.dirname(__file__), "src/iot_lib.so")
lib = ctypes.CDLL(lib_path)

lib.train.restype     = ctypes.c_int
lib.train.argtypes    = [ctypes.c_char_p]
lib.predict.restype   = ctypes.c_double
lib.predict.argtypes  = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.get_mse.restype   = ctypes.c_double
lib.get_weights.argtypes = [ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double)]

# ── App principal ────────────────────────────────────────────────
class IoTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema IoT — Predicción de Temperatura")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e2e")
        self.trained = False
        self.historial = []
        self.build_ui()

    def build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg="#313244", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="🌡 Sistema IoT — Predicción de Temperatura",
                 font=("Courier", 16, "bold"),
                 bg="#313244", fg="#cdd6f4").pack()

        # ── Panel izquierdo ──
        left = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=20)
        left.pack(side="left", fill="y")

        # Botón entrenar
        self.btn_train = tk.Button(left, text="⚙ Entrenar Modelo",
                                   command=self.train_model,
                                   bg="#89b4fa", fg="#1e1e2e",
                                   font=("Courier", 11, "bold"),
                                   width=22, pady=8, relief="flat")
        self.btn_train.pack(pady=(0,10))

        self.lbl_status = tk.Label(left, text="Estado: sin entrenar",
                                   bg="#1e1e2e", fg="#f38ba8",
                                   font=("Courier", 10))
        self.lbl_status.pack(pady=(0,20))

        # Inputs
        campos = [
            ("Humedad interior (%)", "85", "inHumid"),
            ("Temp. exterior (°C)",  "31", "outTemp"),
            ("Batería (V)",          "4.0","battery"),
        ]
        self.entries = {}
        for label, default, key in campos:
            tk.Label(left, text=label, bg="#1e1e2e", fg="#cdd6f4",
                     font=("Courier", 10)).pack(anchor="w")
            e = tk.Entry(left, font=("Courier", 11),
                         bg="#313244", fg="#cdd6f4",
                         insertbackground="white", relief="flat",
                         width=22)
            e.insert(0, default)
            e.pack(pady=(2,10), ipady=4)
            self.entries[key] = e

        # Botón predecir
        self.btn_pred = tk.Button(left, text="🔮 Predecir Temperatura",
                                  command=self.make_prediction,
                                  bg="#a6e3a1", fg="#1e1e2e",
                                  font=("Courier", 11, "bold"),
                                  width=22, pady=8, relief="flat",
                                  state="disabled")
        self.btn_pred.pack(pady=(0,10))

        # Resultado
        self.lbl_result = tk.Label(left, text="",
                                   bg="#1e1e2e", fg="#f9e2af",
                                   font=("Courier", 18, "bold"))
        self.lbl_result.pack(pady=10)

        # MSE
        self.lbl_mse = tk.Label(left, text="",
                                 bg="#1e1e2e", fg="#89dceb",
                                 font=("Courier", 10))
        self.lbl_mse.pack()

        # ── Panel derecho — gráficas ──
        right = tk.Frame(self.root, bg="#1e1e2e")
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Notebook con tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244",
                        foreground="#cdd6f4", padding=[10,5],
                        font=("Courier", 10))
        style.map("TNotebook.Tab", background=[("selected","#89b4fa")],
                  foreground=[("selected","#1e1e2e")])

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1 — historial de predicciones
        self.tab_pred = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(self.tab_pred, text="Predicciones")
        self.fig_pred, self.ax_pred = plt.subplots(figsize=(5,3.5),
                                                    facecolor="#1e1e2e")
        self.ax_pred.set_facecolor("#313244")
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, self.tab_pred)
        self.canvas_pred.get_tk_widget().pack(fill="both", expand=True)

        # Tab 2 — desempeño paralelo
        self.tab_perf = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(self.tab_perf, text="Desempeño Paralelo")
        self.fig_perf, self.ax_perf = plt.subplots(1, 2, figsize=(5,3.5),
                                                     facecolor="#1e1e2e")
        self.canvas_perf = FigureCanvasTkAgg(self.fig_perf, self.tab_perf)
        self.canvas_perf.get_tk_widget().pack(fill="both", expand=True)
        self.plot_performance()

    def train_model(self):
        self.lbl_status.config(text="Entrenando...", fg="#f9e2af")
        self.btn_train.config(state="disabled")
        self.root.update()

        def do_train():
            csv = b"data/18DAYSDATA_corrected.csv"
            n = lib.train(csv)
            if n > 0:
                mse = lib.get_mse()
                self.trained = True
                self.lbl_status.config(
                    text=f"✓ Entrenado ({n:,} muestras)",
                    fg="#a6e3a1")
                self.lbl_mse.config(text=f"MSE: {mse:.6f}")
                self.btn_pred.config(state="normal")
            else:
                self.lbl_status.config(text="Error al entrenar",
                                        fg="#f38ba8")
            self.btn_train.config(state="normal")

        threading.Thread(target=do_train, daemon=True).start()

    def make_prediction(self):
        try:
            inHumid = float(self.entries["inHumid"].get())
            outTemp = float(self.entries["outTemp"].get())
            battery = float(self.entries["battery"].get())
        except ValueError:
            messagebox.showerror("Error", "Ingresa valores numéricos")
            return

        pred = lib.predict(inHumid, outTemp, battery)
        self.lbl_result.config(text=f"inTemp: {pred:.2f}°C")

        # Agregar al historial
        self.historial.append({
            "inHumid": inHumid,
            "outTemp": outTemp,
            "pred":    pred
        })
        self.update_pred_plot()

    def update_pred_plot(self):
        self.ax_pred.clear()
        self.ax_pred.set_facecolor("#313244")
        preds = [h["pred"] for h in self.historial]
        self.ax_pred.plot(preds, "o-", color="#89b4fa", lw=2, markersize=6)
        self.ax_pred.axhline(0, color="#f38ba8", ls="--", alpha=0.5,
                              label="0°C")
        self.ax_pred.axhline(8, color="#fab387", ls="--", alpha=0.5,
                              label="8°C (límite)")
        self.ax_pred.set_title("Historial de Predicciones",
                                color="#cdd6f4", fontsize=10)
        self.ax_pred.set_ylabel("Temp. interior (°C)", color="#cdd6f4")
        self.ax_pred.set_xlabel("Predicción #", color="#cdd6f4")
        self.ax_pred.tick_params(colors="#cdd6f4")
        self.ax_pred.legend(fontsize=8, facecolor="#313244",
                             labelcolor="#cdd6f4")
        self.canvas_pred.draw()

    def plot_performance(self):
        try:
            df = pd.read_csv("results/resultados.csv")
        except:
            return

        for ax in self.ax_perf:
            ax.set_facecolor("#313244")

        # Speedup
        omp = df[df["version"] == "openmp"]
        thr = df[df["version"] == "threads"]
        hilos = [2, 4]

        self.ax_perf[0].plot(hilos, omp["speedup"].values,
                              "o-", color="#a6e3a1", label="OpenMP", lw=2)
        self.ax_perf[0].plot(hilos, thr["speedup"].values,
                              "s-", color="#f38ba8", label="threads", lw=2)
        self.ax_perf[0].plot(hilos, hilos, "k--", alpha=0.4, label="Ideal")
        self.ax_perf[0].set_title("Speedup", color="#cdd6f4", fontsize=10)
        self.ax_perf[0].set_ylabel("Speedup", color="#cdd6f4")
        self.ax_perf[0].tick_params(colors="#cdd6f4")
        self.ax_perf[0].legend(fontsize=8, facecolor="#313244",
                                labelcolor="#cdd6f4")

        # Eficiencia
        self.ax_perf[1].plot(hilos, omp["eficiencia"].values,
                              "o-", color="#a6e3a1", label="OpenMP", lw=2)
        self.ax_perf[1].plot(hilos, thr["eficiencia"].values,
                              "s-", color="#f38ba8", label="threads", lw=2)
        self.ax_perf[1].axhline(1.0, color="white", ls="--",
                                 alpha=0.4, label="Ideal")
        self.ax_perf[1].set_title("Eficiencia", color="#cdd6f4", fontsize=10)
        self.ax_perf[1].tick_params(colors="#cdd6f4")
        self.ax_perf[1].legend(fontsize=8, facecolor="#313244",
                                labelcolor="#cdd6f4")

        self.fig_perf.tight_layout()
        self.canvas_perf.draw()

# ── Iniciar app ──────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = IoTApp(root)
    root.mainloop()
