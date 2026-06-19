# Hardware

Contiene los diagramas de conexión, fotos del prototipo y código del microcontrolador.

## Contenido esperado
- `diagrama_conexion.png` — Esquemático del circuito (Fritzing / Proteus)
- `fotos_prototipo/` — Fotografías del hardware armado
- `firmware/` — Código Arduino/ESP32 para adquisición (si aplica)

## Opciones de adquisición
- **BIOPAC:** Conexión vía API nativa o protocolo LSL
- **ESP32/STM32:** Amplificador de biopotenciales + ADC 12-bit, envío por UART/WiFi
- **DAQ (NI/similar):** Adquisición directa en PC

## Parámetros de adquisición recomendados
| Señal | Fs mínima | Filtro hardware |
|-------|-----------|-----------------|
| ECG   | 500 Hz    | 0.05–150 Hz     |
| EEG   | 256 Hz    | 0.5–50 Hz       |
