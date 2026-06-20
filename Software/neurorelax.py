import sys
import bioread
import numpy as np

from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import welch

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pyqtgraph as pg


# =====================================================
# CONFIGURACIÓN
# =====================================================

FS = 1000

VENTANA_EEG = 5      # segundos
VENTANA_RMSSD = 30   # segundos

UPDATE_MS = 500     # 1 segundo


# =====================================================
# FILTROS
# =====================================================

def bandpass(signal, low, high, fs, order=4):

    nyq = fs / 2

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype='band'
    )

    return filtfilt(b, a, signal)


# =====================================================
# EEG
# =====================================================

def alpha_power(segmento):

    eeg = bandpass(segmento, 1, 40, FS)

    freqs, psd = welch(
        eeg,
        fs=FS,
        nperseg=min(4096, len(eeg))
    )

    mask = (freqs >= 8) & (freqs <= 13)

    alpha = np.trapezoid(
        psd[mask],
        freqs[mask]
    )
    return float(alpha)


# =====================================================
# ECG
# =====================================================

def calcular_rmssd(segmento):

    ecg = bandpass(segmento, 5, 25, FS)

    peaks, _ = find_peaks(
        ecg,
        distance=0.5 * FS,
        prominence=np.std(ecg)
    )

    if len(peaks) < 3:
        return np.nan

    rr = np.diff(peaks) / FS

    rmssd = np.sqrt(
        np.mean(
            np.diff(rr) ** 2
        )
    )

    return rmssd * 1000  # ms


# =====================================================
# CALIBRACIÓN
# =====================================================

def calibrar(path):

    data = bioread.read_file(path)

    eeg = data.channels[0].data
    ecg = data.channels[1].data

    alpha_basal = alpha_power(eeg)

    rmssd_basal = calcular_rmssd(ecg)

    return alpha_basal, rmssd_basal


# =====================================================
# GUI
# =====================================================

class SemaforoGUI(QWidget):

    def __init__(
            self,
            eeg,
            ecg,
            alpha_basal,
            rmssd_basal):

        super().__init__()

        self.eeg = eeg
        self.ecg = ecg

        self.alpha_basal = alpha_basal
        self.rmssd_basal = rmssd_basal

        self.ptr = 0
        # Historial para suavizado
        self.hist_alpha = []
        # Estado actual del semáforo
        self.estado_actual = "🟡 INTERMEDIO"

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(UPDATE_MS)

    # -------------------------------------------------

    def init_ui(self):

        self.setWindowTitle(
            "Semáforo de Relajación"
        )

        layout = QVBoxLayout()

        self.plot_ecg = pg.PlotWidget()
        self.plot_eeg = pg.PlotWidget()

        self.curve_ecg = self.plot_ecg.plot()
        self.curve_eeg = self.plot_eeg.plot()

        layout.addWidget(self.plot_ecg)
        layout.addWidget(self.plot_eeg)

        self.estado = QLabel("CALIBRANDO")
        self.estado.setStyleSheet("""
            font-size:30px;
            font-weight:bold;
        """)

        layout.addWidget(self.estado)

        self.lbl_alpha = QLabel()
        self.lbl_rmssd = QLabel()
        self.lbl_score = QLabel()

        layout.addWidget(self.lbl_alpha)
        layout.addWidget(self.lbl_rmssd)
        layout.addWidget(self.lbl_score)

        self.setLayout(layout)

        self.resize(1200, 800)

    # -------------------------------------------------

    def update_data(self):

        self.ptr += int(FS * UPDATE_MS / 1000)

        if self.ptr >= len(self.eeg):
            self.timer.stop()
            return

        # -----------------------------------------
        # EEG
        # -----------------------------------------

        eeg_ini = max(
            0,
            self.ptr - VENTANA_EEG * FS
        )

        eeg_seg = self.eeg[eeg_ini:self.ptr]

        alpha = alpha_power(eeg_seg)

        # -----------------------------------------
        # ECG
        # -----------------------------------------

        ecg_ini = max(
            0,
            self.ptr - VENTANA_RMSSD * FS
        )

        ecg_seg = self.ecg[ecg_ini:self.ptr]

        rmssd = calcular_rmssd(ecg_seg)

        if np.isnan(rmssd):
            rmssd = self.rmssd_basal

        # -----------------------------------------
        # Semaforo basado en Alpha
        # -----------------------------------------

        alpha_ratio = alpha / self.alpha_basal
        self.hist_alpha.append(alpha_ratio)
        if len(self.hist_alpha) > 10:
            self.hist_alpha.pop(0)
        alpha_ratio_suave = np.mean(self.hist_alpha)
        reduccion = (1 - alpha_ratio_suave) * 100

        if self.estado_actual == "🟢 RELAJADO":
            if alpha_ratio_suave < 0.60:
                self.estado_actual = "🟡 INTERMEDIO"
        elif self.estado_actual == "🟡 INTERMEDIO":
            if alpha_ratio_suave >= 0.90:
                self.estado_actual = "🟢 RELAJADO"
            elif alpha_ratio_suave < 0.40:
                self.estado_actual = "🔴 ESTRÉS"
        elif self.estado_actual == "🔴 ESTRÉS":
            if alpha_ratio_suave > 0.55:
                self.estado_actual = "🟡 INTERMEDIO"
        estado = self.estado_actual

        print(
            f"Alpha={alpha:.2f} | "
            f"Ratio={alpha_ratio:.2f} | "
            f"Suave={alpha_ratio_suave:.2f} | "
            f"Estado={estado}"
        )

        self.estado.setText(estado)

        # -----------------------------------------
        # LABELS
        # -----------------------------------------

        self.lbl_alpha.setText(
            f"Alpha actual: {alpha:.2f}"
        )

        self.lbl_rmssd.setText(
            f"RMSSD: {rmssd:.2f} ms"
        )

        self.lbl_score.setText(
            f"Alpha normalizada: {alpha_ratio_suave:.2f} | Reducción: {reduccion:.1f}%"
        )

        # -----------------------------------------
        # PLOTS
        # -----------------------------------------

        self.curve_ecg.setData(
            ecg_seg[-5000:]
        )

        self.curve_eeg.setData(
            eeg_seg
        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    calibracion = "calibracion_Sergio_2.acq"

    prueba = "prueba_reposo_estres_Sergio_3_2minutos.acq"

    print("Calculando calibración...")

    alpha_basal, rmssd_basal = calibrar(
        calibracion
    )

    print(
        "Alpha basal:",
        alpha_basal
    )

    print(
        "RMSSD basal:",
        rmssd_basal
    )

    data = bioread.read_file(prueba)

    eeg = data.channels[0].data
    ecg = data.channels[1].data

    app = QApplication(sys.argv)

    gui = SemaforoGUI(
        eeg,
        ecg,
        alpha_basal,
        rmssd_basal
    )

    gui.show()

    sys.exit(app.exec_())
