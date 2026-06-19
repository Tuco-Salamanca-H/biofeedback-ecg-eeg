"""
main_gui.py — Sistema de Biofeedback para Control del Estrés (ECG + EEG)
Proyecto Final — Procesamiento de Señales Biomédicas

Descripción:
    Interfaz gráfica en tiempo real que adquiere señales de ECG y EEG,
    extrae características en tiempo y frecuencia, y muestra el
    "Semáforo de Relajación".

Autores: Equipo Biofeedback
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QGroupBox, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from signal_processing import ECGProcessor, EEGProcessor
from data_acquisition import DataAcquisition


class MainWindow(QMainWindow):
    """Ventana principal del sistema de biofeedback."""

    FS_ECG = 500       # Hz
    FS_EEG = 256       # Hz
    WINDOW_SEC = 2     # segundos de buffer deslizante
    UPDATE_RATE_MS = 200  # ms → 5 veces por segundo

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biofeedback — Semáforo de Relajación (ECG + EEG)")
        self.setMinimumSize(1200, 700)

        # Buffers deslizantes
        self.ecg_buffer = np.zeros(int(self.FS_ECG * self.WINDOW_SEC))
        self.eeg_buffer = np.zeros(int(self.FS_EEG * self.WINDOW_SEC))

        # Procesadores de señal
        self.ecg_proc = ECGProcessor(fs=self.FS_ECG)
        self.eeg_proc = EEGProcessor(fs=self.FS_EEG)

        # Adquisición de datos
        self.daq = DataAcquisition(fs_ecg=self.FS_ECG, fs_eeg=self.FS_EEG)

        self._build_ui()

        # Timer de actualización (≥5 Hz según especificaciones)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(self.UPDATE_RATE_MS)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Panel izquierdo: gráficas ---
        left = QVBoxLayout()

        # Gráfica ECG
        self.fig_ecg = Figure(figsize=(6, 2), tight_layout=True)
        self.ax_ecg = self.fig_ecg.add_subplot(111)
        self.ax_ecg.set_title("ECG — Derivación II")
        self.ax_ecg.set_xlabel("Tiempo (s)")
        self.ax_ecg.set_ylabel("Amplitud (mV)")
        self.canvas_ecg = FigureCanvas(self.fig_ecg)
        left.addWidget(self.canvas_ecg)

        # Gráfica EEG
        self.fig_eeg = Figure(figsize=(6, 2), tight_layout=True)
        self.ax_eeg = self.fig_eeg.add_subplot(111)
        self.ax_eeg.set_title("EEG — Banda Alfa (8–13 Hz)")
        self.ax_eeg.set_xlabel("Tiempo (s)")
        self.ax_eeg.set_ylabel("Amplitud (µV)")
        self.canvas_eeg = FigureCanvas(self.fig_eeg)
        left.addWidget(self.canvas_eeg)

        # Gráfica PSD
        self.fig_psd = Figure(figsize=(6, 2), tight_layout=True)
        self.ax_psd = self.fig_psd.add_subplot(111)
        self.ax_psd.set_title("PSD — EEG (FFT)")
        self.ax_psd.set_xlabel("Frecuencia (Hz)")
        self.ax_psd.set_ylabel("Potencia (µV²/Hz)")
        self.canvas_psd = FigureCanvas(self.fig_psd)
        left.addWidget(self.canvas_psd)

        main_layout.addLayout(left, stretch=3)

        # --- Panel derecho: métricas y semáforo ---
        right = QVBoxLayout()

        # Semáforo
        semaforo_box = QGroupBox("Semáforo de Relajación")
        semaforo_box.setFont(QFont("Arial", 12, QFont.Bold))
        sem_layout = QVBoxLayout(semaforo_box)
        self.semaforo_label = QLabel("●")
        self.semaforo_label.setAlignment(Qt.AlignCenter)
        self.semaforo_label.setFont(QFont("Arial", 80))
        self.semaforo_label.setStyleSheet("color: gray;")
        self.estado_label = QLabel("Inicializando...")
        self.estado_label.setAlignment(Qt.AlignCenter)
        self.estado_label.setFont(QFont("Arial", 14))
        sem_layout.addWidget(self.semaforo_label)
        sem_layout.addWidget(self.estado_label)
        right.addWidget(semaforo_box)

        # Métricas ECG
        ecg_box = QGroupBox("Métricas ECG — HRV")
        ecg_grid = QGridLayout(ecg_box)
        ecg_grid.addWidget(QLabel("FC (BPM):"), 0, 0)
        self.bpm_label = QLabel("--")
        ecg_grid.addWidget(self.bpm_label, 0, 1)
        ecg_grid.addWidget(QLabel("RMSSD (ms):"), 1, 0)
        self.rmssd_label = QLabel("--")
        ecg_grid.addWidget(self.rmssd_label, 1, 1)
        ecg_grid.addWidget(QLabel("LF/HF:"), 2, 0)
        self.lfhf_label = QLabel("--")
        ecg_grid.addWidget(self.lfhf_label, 2, 1)
        right.addWidget(ecg_box)

        # Métricas EEG
        eeg_box = QGroupBox("Métricas EEG — Alfa")
        eeg_grid = QGridLayout(eeg_box)
        eeg_grid.addWidget(QLabel("Potencia Alfa (µV²):"), 0, 0)
        self.alpha_label = QLabel("--")
        eeg_grid.addWidget(self.alpha_label, 0, 1)
        right.addWidget(eeg_box)

        # Botón inicio/parada
        self.btn_toggle = QPushButton("⏸ Pausar")
        self.btn_toggle.setFont(QFont("Arial", 12))
        self.btn_toggle.clicked.connect(self._toggle)
        right.addWidget(self.btn_toggle)

        right.addStretch()
        main_layout.addLayout(right, stretch=1)

    # ------------------------------------------------------------------
    def _update(self):
        """Callback del timer: adquiere datos, procesa y actualiza GUI."""
        # 1. Adquirir nuevas muestras
        new_ecg, new_eeg = self.daq.get_samples()

        # 2. Actualizar buffers deslizantes
        n_ecg = len(new_ecg)
        n_eeg = len(new_eeg)
        self.ecg_buffer = np.roll(self.ecg_buffer, -n_ecg)
        self.ecg_buffer[-n_ecg:] = new_ecg
        self.eeg_buffer = np.roll(self.eeg_buffer, -n_eeg)
        self.eeg_buffer[-n_eeg:] = new_eeg

        # 3. Procesar ECG
        bpm, rmssd, lf_hf = self.ecg_proc.process(self.ecg_buffer)

        # 4. Procesar EEG
        eeg_alpha, alpha_power, freqs, psd = self.eeg_proc.process(self.eeg_buffer)

        # 5. Actualizar etiquetas
        self.bpm_label.setText(f"{bpm:.1f}" if bpm else "--")
        self.rmssd_label.setText(f"{rmssd:.2f}" if rmssd else "--")
        self.lfhf_label.setText(f"{lf_hf:.3f}" if lf_hf else "--")
        self.alpha_label.setText(f"{alpha_power:.4f}" if alpha_power else "--")

        # 6. Lógica del semáforo
        self._update_semaforo(alpha_power, lf_hf)

        # 7. Actualizar gráficas
        t_ecg = np.linspace(0, self.WINDOW_SEC, len(self.ecg_buffer))
        self.ax_ecg.clear()
        self.ax_ecg.plot(t_ecg, self.ecg_buffer, "b", linewidth=0.8)
        self.ax_ecg.set_title("ECG — Derivación II")
        self.ax_ecg.set_xlabel("Tiempo (s)")
        self.canvas_ecg.draw()

        t_eeg = np.linspace(0, self.WINDOW_SEC, len(self.eeg_buffer))
        self.ax_eeg.clear()
        self.ax_eeg.plot(t_eeg, eeg_alpha, "g", linewidth=0.8)
        self.ax_eeg.set_title("EEG — Banda Alfa filtrada")
        self.canvas_eeg.draw()

        if freqs is not None and psd is not None:
            self.ax_psd.clear()
            self.ax_psd.plot(freqs, psd, "r", linewidth=0.8)
            self.ax_psd.axvspan(8, 13, alpha=0.2, color="green", label="Alfa")
            self.ax_psd.set_xlim(0, 50)
            self.ax_psd.set_title("PSD — EEG (FFT)")
            self.ax_psd.legend()
            self.canvas_psd.draw()

    def _update_semaforo(self, alpha_power, lf_hf):
        """Actualiza el color del semáforo según las métricas."""
        if alpha_power is None or lf_hf is None:
            self.semaforo_label.setStyleSheet("color: gray;")
            self.estado_label.setText("Adquiriendo señal...")
            return

        # Umbrales (ajustables según calibración del sujeto)
        ALPHA_THRESHOLD = 0.5   # µV² — umbral de potencia alfa
        LFHF_THRESHOLD = 2.0    # relación LF/HF

        if alpha_power > ALPHA_THRESHOLD and lf_hf < LFHF_THRESHOLD:
            self.semaforo_label.setStyleSheet("color: #00cc00;")
            self.estado_label.setText("🟢 RELAJADO")
        else:
            self.semaforo_label.setStyleSheet("color: #cc0000;")
            self.estado_label.setText("🔴 ESTRESADO")

    def _toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_toggle.setText("▶ Reanudar")
        else:
            self.timer.start(self.UPDATE_RATE_MS)
            self.btn_toggle.setText("⏸ Pausar")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
