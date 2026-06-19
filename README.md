# Biofeedback para Control del Estrés (ECG + EEG)

## Descripción
Sistema de procesamiento multitarea en tiempo real que adquiere, procesa y fusiona señales de ECG y EEG para generar un indicador automatizado de nivel de estrés ("Semáforo de Relajación").

## Requisitos del sistema
- Python 3.9+
- Librerías necesarias:

```bash
pip install numpy scipy matplotlib pyqt5 pylsl neurokit2
```

## Estructura del repositorio
```
/Hardware   → Diagramas de conexión y código de microcontrolador
/Software   → Scripts principales de la GUI y funciones auxiliares
/Reporte    → Artículo IEEE en PDF y .docx
README.md   → Este archivo
```

## Cómo ejecutar
1. Conectar hardware (BIOPAC vía LSL o ESP32/STM32 vía DAQ)
2. Instalar dependencias: `pip install -r Software/requirements.txt`
3. Ejecutar la interfaz: `python Software/main_gui.py`

## Señales procesadas
- **ECG (derivación II):** Detección de picos R, tacograma R-R, HRV (RMSSD, LF/HF)
- **EEG (Fz/Cz):** Filtro banda Alfa (8–13 Hz), PSD via FFT

## Lógica del Semáforo de Relajación
| Estado | Condición |
|--------|-----------|
| 🟢 Verde (Relajado) | Potencia Alfa ↑ y LF/HF ↓ |
| 🔴 Rojo (Estresado) | Potencia Alfa ↓ y LF/HF ↑ |


## Autores
Humberto Alonso Rodríguez Paredes,
Sergio Daniel Salamanca Hernández

