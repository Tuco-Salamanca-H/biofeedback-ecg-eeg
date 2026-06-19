"""
data_acquisition.py — Módulo de adquisición de datos

Soporta:
  - Modo simulación (por defecto): genera señales sintéticas para pruebas
  - LSL (Lab Streaming Layer): conexión a BIOPAC u otros sistemas clínicos
  - Serial (ESP32/STM32): lectura por puerto COM

Cambiar MODE para seleccionar la fuente de datos.
"""

import numpy as np

MODE = "simulation"   # Opciones: "simulation" | "lsl" | "serial"


class DataAcquisition:
    """Clase de adquisición de datos multi-modo."""

    def __init__(self, fs_ecg=500, fs_eeg=256):
        self.fs_ecg = fs_ecg
        self.fs_eeg = fs_eeg
        self.t = 0.0
        self.samples_per_call_ecg = int(fs_ecg * 0.2)  # 200 ms de datos
        self.samples_per_call_eeg = int(fs_eeg * 0.2)

        if MODE == "lsl":
            self._init_lsl()
        elif MODE == "serial":
            self._init_serial()

    # ------------------------------------------------------------------
    # Modo simulación — genera ECG y EEG sintéticos con ruido
    # ------------------------------------------------------------------
    def _sim_ecg(self, n):
        """Simula una derivación II de ECG a fs_ecg Hz."""
        t = np.linspace(self.t, self.t + n/self.fs_ecg, n)
        # Componente sinusoidal base + picos R sintéticos
        ecg = 0.05 * np.sin(2 * np.pi * 1.2 * t)
        for pk in t[::int(self.fs_ecg * 0.833)]:  # ~72 BPM
            idx = np.argmin(np.abs(t - pk))
            if idx < n:
                ecg[idx] += 1.0  # pico R
        ecg += 0.01 * np.random.randn(n)
        return ecg

    def _sim_eeg(self, n):
        """Simula un canal EEG con actividad alfa y ruido."""
        t = np.linspace(self.t, self.t + n/self.fs_eeg, n)
        eeg = 20 * np.sin(2 * np.pi * 10 * t)   # 10 Hz — alfa
        eeg += 5  * np.sin(2 * np.pi * 25 * t)   # beta
        eeg += 2  * np.random.randn(n)             # ruido
        return eeg

    # ------------------------------------------------------------------
    # LSL
    # ------------------------------------------------------------------
    def _init_lsl(self):
        from pylsl import StreamInlet, resolve_stream
        print("Buscando streams LSL...")
        streams_ecg = resolve_stream("type", "ECG")
        streams_eeg = resolve_stream("type", "EEG")
        self.inlet_ecg = StreamInlet(streams_ecg[0])
        self.inlet_eeg = StreamInlet(streams_eeg[0])

    def _get_lsl(self):
        samples_ecg, _ = self.inlet_ecg.pull_chunk(max_samples=self.samples_per_call_ecg)
        samples_eeg, _ = self.inlet_eeg.pull_chunk(max_samples=self.samples_per_call_eeg)
        ecg = np.array(samples_ecg).flatten() if samples_ecg else np.zeros(self.samples_per_call_ecg)
        eeg = np.array(samples_eeg).flatten() if samples_eeg else np.zeros(self.samples_per_call_eeg)
        return ecg, eeg

    # ------------------------------------------------------------------
    # Serial
    # ------------------------------------------------------------------
    def _init_serial(self):
        import serial
        self.ser = serial.Serial("COM3", baudrate=115200, timeout=0.1)

    def _get_serial(self):
        # Protocolo simple: línea "ECG,EEG\\n"
        ecg, eeg = [], []
        for _ in range(self.samples_per_call_ecg):
            line = self.ser.readline().decode(errors="ignore").strip()
            try:
                parts = line.split(",")
                ecg.append(float(parts[0]))
                eeg.append(float(parts[1]))
            except Exception:
                ecg.append(0.0)
                eeg.append(0.0)
        return np.array(ecg), np.array(eeg)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def get_samples(self):
        """Retorna (ecg_samples, eeg_samples) según el modo configurado."""
        if MODE == "simulation":
            ecg = self._sim_ecg(self.samples_per_call_ecg)
            eeg = self._sim_eeg(self.samples_per_call_eeg)
            self.t += 0.2
            return ecg, eeg
        elif MODE == "lsl":
            return self._get_lsl()
        elif MODE == "serial":
            return self._get_serial()
        return np.zeros(self.samples_per_call_ecg), np.zeros(self.samples_per_call_eeg)
