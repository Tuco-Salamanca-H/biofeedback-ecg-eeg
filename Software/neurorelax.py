mport sys
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

# Paleta de colores oscura estilo neuromonitor
COLOR_BG        = "#0d1117"
COLOR_PANEL     = "#161b22"
COLOR_BORDER    = "#30363d"
COLOR_TEXT      = "#e6edf3"
COLOR_SUBTEXT   = "#8b949e"
COLOR_ACCENT    = "#58a6ff"
COLOR_GREEN     = "#3fb950"
COLOR_YELLOW    = "#d29922"
COLOR_RED       = "#f85149"
COLOR_PLOT_EEG  = "#58a6ff"
COLOR_PLOT_ECG  = "#3fb950"


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
        self.hist_alpha = []
        self.estado_actual = "🟡 INTERMEDIO"

        self.elapsed_seconds = 0
        self.total_seconds = len(eeg) // FS

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(UPDATE_MS)

        self.crono_timer = QTimer()
        self.crono_timer.timeout.connect(self._tick_crono)
        self.crono_timer.start(1000)

    def _tick_crono(self):
        if self.elapsed_seconds < self.total_seconds:
            self.elapsed_seconds += 1
            self._update_crono_label()

    def _fmt_tiempo(self, segundos):
        m = segundos // 60
        s = segundos % 60
        return f"{m:02d}:{s:02d}"

    def _update_crono_label(self):
        elapsed_str = self._fmt_tiempo(self.elapsed_seconds)
        total_str   = self._fmt_tiempo(self.total_seconds)
        self.lbl_crono.setText(f"⏱  {elapsed_str} / {total_str}")

    def init_ui(self):

        self.setWindowTitle("Semáforo de Relajación")
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEXT};")

        root = QHBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        lbl_plots = QLabel("SEÑALES EN TIEMPO REAL")
        lbl_plots.setStyleSheet(f"color: {COLOR_SUBTEXT}; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        left.addWidget(lbl_plots)

        ecg_label = QLabel("ECG")
        ecg_label.setStyleSheet(f"color: {COLOR_PLOT_ECG}; font-size: 12px; font-weight: bold;")
        left.addWidget(ecg_label)

        self.plot_ecg = pg.PlotWidget()
        self._style_plot(self.plot_ecg, COLOR_PLOT_ECG)
        self.curve_ecg = self.plot_ecg.plot(pen=pg.mkPen(COLOR_PLOT_ECG, width=1.5))
        left.addWidget(self.plot_ecg)

        eeg_label = QLabel("EEG")
        eeg_label.setStyleSheet(f"color: {COLOR_PLOT_EEG}; font-size: 12px; font-weight: bold;")
        left.addWidget(eeg_label)

        self.plot_eeg = pg.PlotWidget()
        self._style_plot(self.plot_eeg, COLOR_PLOT_EEG)
        self.curve_eeg = self.plot_eeg.plot(pen=pg.mkPen(COLOR_PLOT_EEG, width=1.5))
        left.addWidget(self.plot_eeg)

        right = QVBoxLayout()
        right.setSpacing(12)
        right.setAlignment(Qt.AlignTop)

        crono_panel = self._make_panel()
        crono_layout = QVBoxLayout(crono_panel)
        crono_layout.setContentsMargins(16, 12, 16, 12)
        crono_title = QLabel("DURACIÓN")
        crono_title.setStyleSheet(f"color: {COLOR_SUBTEXT}; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        crono_layout.addWidget(crono_title)
        self.lbl_crono = QLabel(f"⏱  00:00 / {self._fmt_tiempo(self.total_seconds)}")
        self.lbl_crono.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 26px; font-weight: bold; font-family: monospace;")
        crono_layout.addWidget(self.lbl_crono)
        right.addWidget(crono_panel)

        estado_panel = self._make_panel()
        estado_layout = QVBoxLayout(estado_panel)
        estado_layout.setContentsMargins(16, 12, 16, 12)
        estado_title = QLabel("ESTADO")
        estado_title.setStyleSheet(f"color: {COLOR_SUBTEXT}; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        estado_layout.addWidget(estado_title)
        self.estado = QLabel("CALIBRANDO")
        self.estado.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLOR_YELLOW};")
        estado_layout.addWidget(self.estado)
        right.addWidget(estado_panel)

        metrics_panel = self._make_panel()
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(16, 12, 16, 12)
        metrics_layout.setSpacing(10)
        metrics_title = QLabel("MÉTRICAS")
        metrics_title.setStyleSheet(f"color: {COLOR_SUBTEXT}; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        metrics_layout.addWidget(metrics_title)
        self.lbl_alpha = QLabel("Alpha actual: —")
        self.lbl_rmssd = QLabel("RMSSD: —")
        self.lbl_score = QLabel("Alpha normalizada: — | Reducción: —")
        for lbl in [self.lbl_alpha, self.lbl_rmssd, self.lbl_score]:
            lbl.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 14px; padding: 4px 0px;")
            lbl.setWordWrap(True)
            metrics_layout.addWidget(lbl)
        right.addWidget(metrics_panel)
        right.addStretch()

        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=1)
        self.setLayout(root)
        self.resize(1300, 800)

    def _make_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; }}")
        return panel

    def _style_plot(self, plot_widget, accent_color):
        plot_widget.setBackground(COLOR_PANEL)
        plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(COLOR_BORDER))
        plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(COLOR_BORDER))
        plot_widget.getPlotItem().getAxis('left').setTextPen(pg.mkPen(COLOR_SUBTEXT))
        plot_widget.getPlotItem().getAxis('bottom').setTextPen(pg.mkPen(COLOR_SUBTEXT))
        plot_widget.showGrid(x=True, y=True, alpha=0.15)
        plot_widget.setMinimumHeight(220)

    def update_data(self):

        self.ptr += int(FS * UPDATE_MS / 1000)

        if self.ptr >= len(self.eeg):
            self.timer.stop()
            self.crono_timer.stop()
            self.lbl_crono.setText(f"⏱  {self._fmt_tiempo(self.total_seconds)} / {self._fmt_tiempo(self.total_seconds)}")
            return

        eeg_ini = max(0, self.ptr - VENTANA_EEG * FS)
        eeg_seg = self.eeg[eeg_ini:self.ptr]
        alpha = alpha_power(eeg_seg)

        ecg_ini = max(0, self.ptr - VENTANA_RMSSD * FS)
        ecg_seg = self.ecg[ecg_ini:self.ptr]
        rmssd = calcular_rmssd(ecg_seg)

        if np.isnan(rmssd):
            rmssd = self.rmssd_basal

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

        if "RELAJADO" in estado:
            color_estado = COLOR_GREEN
        elif "ESTRÉS" in estado:
            color_estado = COLOR_RED
        else:
            color_estado = COLOR_YELLOW

        self.estado.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color_estado};")
        self.estado.setText(estado)

        self.lbl_alpha.setText(f"Alpha actual: {alpha:.2f}")
        self.lbl_rmssd.setText(f"RMSSD: {rmssd:.2f} ms")
        self.lbl_score.setText(f"Alpha normalizada: {alpha_ratio_suave:.2f} | Reducción: {reduccion:.1f}%")

        self.curve_ecg.setData(ecg_seg[-5000:])
        self.curve_eeg.setData(eeg_seg)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    calibracion = "LankyCalibracion.acq"
    prueba = "LankyRE.acq"

    print("Calculando calibración...")
    alpha_basal, rmssd_basal = calibrar(calibracion)
    print("Alpha basal:", alpha_basal)
    print("RMSSD basal:", rmssd_basal)

    data = bioread.read_file(prueba)
    eeg = data.channels[0].data
    ecg = data.channels[1].data

    app = QApplication(sys.argv)
    gui = SemaforoGUI(eeg, ecg, alpha_basal, rmssd_basal)
    gui.show()
    sys.exit(app.exec_())
