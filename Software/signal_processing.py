"""
signal_processing.py — Módulo de procesamiento de señales ECG y EEG

Implementa:
  - ECGProcessor: detección de picos R, HRV (RMSSD, LF/HF)
  - EEGProcessor: filtro banda alfa (8-13 Hz), PSD via FFT
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks


class ECGProcessor:
    """
    Procesador de señal ECG.

    Extrae:
    - Frecuencia cardíaca (BPM)
    - RMSSD: raíz cuadrática media de diferencias sucesivas R-R
      RMSSD = sqrt( (1/N-1) * sum( (RR[i+1] - RR[i])^2 ) )
    - Relación LF/HF de la HRV espectral
    """

    def __init__(self, fs=500):
        self.fs = fs
        self._design_filters()

    def _design_filters(self):
        """Filtro pasa-banda 0.5–40 Hz para ECG."""
        nyq = self.fs / 2
        self.b_ecg, self.a_ecg = butter(4, [0.5/nyq, 40/nyq], btype="band")

    def process(self, ecg_raw):
        """
        Procesa ventana de ECG y retorna (BPM, RMSSD, LF/HF).
        Retorna (None, None, None) si no hay suficientes picos R.
        """
        # 1. Filtrado
        ecg_f = filtfilt(self.b_ecg, self.a_ecg, ecg_raw)

        # 2. Detección de picos R (umbral adaptativo)
        threshold = 0.6 * np.max(ecg_f)
        min_distance = int(0.3 * self.fs)  # mínimo 300 ms entre latidos
        peaks, _ = find_peaks(ecg_f, height=threshold, distance=min_distance)

        if len(peaks) < 3:
            return None, None, None

        # 3. Tacograma (intervalos R-R en ms)
        rr_intervals = np.diff(peaks) / self.fs * 1000  # ms

        # 4. BPM instantáneo
        bpm = 60000 / np.mean(rr_intervals)

        # 5. RMSSD
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

        # 6. LF/HF mediante PSD del tacograma (Welch)
        lf_hf = self._compute_lf_hf(rr_intervals)

        return bpm, rmssd, lf_hf

    def _compute_lf_hf(self, rr_ms):
        """
        Calcula relación LF/HF del tacograma R-R.
        LF: 0.04–0.15 Hz  (actividad simpática/parasimpática mixta)
        HF: 0.15–0.40 Hz  (actividad parasimpática - respiración)
        """
        if len(rr_ms) < 4:
            return None

        # Señal R-R en unidades de tiempo real (re-muestreada a 4 Hz)
        fs_rr = 4.0
        t_rr = np.cumsum(rr_ms) / 1000  # segundos
        t_uniform = np.arange(t_rr[0], t_rr[-1], 1/fs_rr)
        rr_interp = np.interp(t_uniform, t_rr, rr_ms)

        if len(rr_interp) < 8:
            return None

        freqs, psd = welch(rr_interp, fs=fs_rr, nperseg=min(len(rr_interp), 64))

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)

        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])

        if hf_power < 1e-10:
            return None

        return lf_power / hf_power


class EEGProcessor:
    """
    Procesador de señal EEG.

    Extrae:
    - Señal filtrada en banda alfa (8–13 Hz)
    - Densidad de potencia espectral (PSD) via FFT (Welch)
    - Potencia total en banda alfa
    """

    def __init__(self, fs=256):
        self.fs = fs
        self._design_filters()

    def _design_filters(self):
        """Filtro pasa-banda Alfa 8–13 Hz (Butterworth orden 4)."""
        nyq = self.fs / 2
        self.b_alpha, self.a_alpha = butter(4, [8/nyq, 13/nyq], btype="band")

    def process(self, eeg_raw):
        """
        Procesa ventana de EEG.
        Retorna (eeg_alpha, alpha_power, freqs, psd).
        """
        # 1. Filtro alfa
        eeg_alpha = filtfilt(self.b_alpha, self.a_alpha, eeg_raw)

        # 2. PSD via Welch (estimación espectral robusta)
        nperseg = min(len(eeg_raw), int(self.fs))
        freqs, psd = welch(eeg_raw, fs=self.fs, nperseg=nperseg)

        # 3. Potencia en banda alfa
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])

        return eeg_alpha, alpha_power, freqs, psd
