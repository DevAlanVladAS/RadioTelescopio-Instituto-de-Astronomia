# Sistema de Adquisición y Procesamiento Espectral para Radiotelescopio (Línea HI - 21 cm)

Desarrollo de un pipeline de software para la captura en tiempo real, procesamiento digital de señales (DSP), análisis astrofísico y visualización espectral de la emisión de hidrógeno neutro (HI a 1420.40575 MHz). 

## Instalación en Windows

Desde PowerShell, en la raíz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inicia la interfaz con `python main.py --ui`; el modo inicial es simulación.
Elige `GNU Radio / HackRF` para que el botón inicie también el flowgraph.

Esto instala PyQt6, PyQtGraph, NumPy, SciPy y pyzmq. GNU Radio y el soporte
SoapySDR/HackRF se instalan por separado en un entorno compatible con GNU Radio;
no se resuelven instalando PyQt6 con pip. Si GNU Radio está en otro entorno,
define `GNURADIO_PYTHON` con la ruta a su `python.exe` antes de iniciar la UI.
Por ejemplo, en PowerShell:

```powershell
$env:GNURADIO_PYTHON = "C:\ruta\al\entorno-gnuradio\python.exe"
python main.py --ui --mode live
```

El worker de `src/ui/qt_bridge.py` puede iniciar/detener el flowgraph en modo
`live`. Los cambios de tamaño FFT e integración se aplican al siguiente inicio,
por lo que hay que detener la adquisición antes de cambiarlos. La calibración de
50 Ohms se mantiene separada por tamaño FFT; si cambias la resolución, calibra
ese tamaño antes de usar una referencia de ruido.

Este proyecto se desarrolla en colaboración con el **Instituto de Astronomía de la UNAM** y **Astrolab**, con aplicación directa en divulgación científica e instrumentación astronómica para la **Noche de las Estrellas** y **Universum, Museo de las Ciencias**.

## Descripción Técnica

El sistema reemplaza las etapas tradicionales de detección analógica posterior al acondicionamiento de radiofrecuencia (RF) mediante una arquitectura basada en Radio Definida por Software (SDR). El flujo de datos abarca desde la lectura de muestras complejas en cuadratura ($I/Q$) hasta la estimación de parámetros cinemáticos de nubes de gas galáctico mediante el Efecto Doppler.

### Módulos Principales

1. **Adquisición y Procesamiento en Banda Base (SDR / DSP):**
   * Recepción de muestras digitales provenientes de un transceptor **HackRF One** e integración con flujos de procesamiento en **GNU Radio**.
   * Cálculo de la Densidad Espectral de Potencia (PSD) mediante Transformada Rápida de Fourier (FFT) e integración temporal para incrementar la Relación Señal a Ruido (SNR).
2. **Análisis Astrofísico:**
   * Conversión del dominio de frecuencias al dominio de velocidades radiales ($v_r$) respecto a la frecuencia en reposo de la transición hiperfina del hidrógeno neutro ($f_0 = 1420.40575\text{ MHz}$).
   * Estimación de flujo integrado y extracción de métricas de línea espectral.
3. **Visualización Científica e Interfaz de Usuario:**
   * Despliegue en tiempo real del espectro de potencia y espectrograma histórico (cascada / *waterfall*).
   * Soporte para operación con hardware en vivo y modo de simulación mediante lecturas pregrabadas para entornos de prueba y exhibición.

## Arquitectura del Proyecto

```text
radiotelescopio-ia/
├── gnuradio/                # Flujos de procesamiento (.grc) y bloques embebidos en Python
├── data_samples/            # Espectros preprocesados (.npz) para pruebas sin hardware
├── src/
│   ├── contracts.py         # Definición de estructuras de datos e interfaces entre módulos
│   ├── acquisition/         # Control de hardware HackRF, comunicación ZMQ y modo simulación
│   ├── physics/             # Algoritmos de conversión Doppler, velocidades radiales y flujo
│   └── ui/                  # Interfaz gráfica y renderizado de series espectrales
├── tests/                   # Pruebas unitarias de procesamiento numérico
├── main.py                  # Punto de entrada de la aplicación
└── requirements.txt         # Dependencias de Python
