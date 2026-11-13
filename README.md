# Sistema de Adquisición y Procesamiento Espectral para Radiotelescopio (Línea HI - 21 cm)

Desarrollo de un pipeline de software para la captura en tiempo real, procesamiento digital de señales (DSP), análisis astrofísico, sonificación y visualización espectral de la emisión de hidrógeno neutro (HI a 1420.40575 MHz).

Este proyecto se desarrolla en colaboración con el **Instituto de Astronomía de la UNAM** y **Astrolab**, con aplicación directa en divulgación científica e instrumentación astronómica para la **Noche de las Estrellas** y **Universum, Museo de las Ciencias**.

---

## Instalación y Configuración

### 1. Ubuntu / Debian (Recomendado para operación con HackRF One)

En Linux, **GNU Radio** y **SoapySDR** están compilados contra el Python del sistema. Para que tanto el flowgraph como la interfaz gráfica compartan acceso sin fricciones:

1. **Instalar dependencias del sistema, drivers del HackRF y PortAudio (para sonificación):**
   ```bash
   sudo apt update
   sudo apt install -y gnuradio hackrf libhackrf-dev soapysdr-tools \
       soapysdr-module-hackrf python3-soapysdr python3-pip python3-venv \
       libportaudio2 libxcb-cursor0
   ```

2. **Crear el entorno virtual heredando los paquetes de GNU Radio (`--system-site-packages`):**
   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verificar que el HackRF One es detectado y no está bloqueado por el kernel:**
   ```bash
   hackrf_info
   ```
   > **Solución al error `Resource busy (-1000)` en Linux:** Si al conectar el HackRF el kernel de Linux carga automáticamente el módulo DVB/V4L2 `hackrf` y bloquea el dispositivo, libéralo ejecutando:
   > ```bash
   > sudo modprobe -r hackrf
   > ```

---

### 2. Instalación en Windows

Desde PowerShell, en la raíz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(Nota: En Windows, GNU Radio y el soporte SoapySDR/HackRF se instalan por separado mediante Radioconda).*

---

## Guía de Uso (`RADIOASTROLAB`)

La ventana principal (`src/ui/main_window.py`) integra la recepción ZMQ, el cálculo astrofísico, el buffer de integración temporal, la sonificación y el despliegue gráfico. **No es necesario ejecutar `main.py` al usar la interfaz gráfica.**

### Opción A: Modo Simulación y Grabación Previa (`.npz`) — *1 sola terminal*

Ideal para demostraciones en entornos con alta interferencia de radiofrecuencia (como la **Noche de las Estrellas**) o pruebas sin hardware conectado:

```bash
python -m src.ui.main_window
```

Dentro de la interfaz puedes:
* Seleccionar **`Grabación Noche de las Estrellas (.npz)`** en el menú `Escenario:` para reproducir datos reales previamente capturados en `data_samples/`.
* Elegir cualquiera de los escenarios astrofísicos preconfigurados (nubes acercándose, brazo espiral alejándose, hidrógeno en reposo) o ajustar los deslizadores de **Velocidad**, **Pico HI** y **Ruido** en tiempo real.
* Modificar el **Tiempo de integración (`1 s`, `5 s`, `15 s`)** para observar cómo el buffer circular reduce la varianza del piso de ruido térmico.
* Activar **`🔊 Sonido: ON`** para escuchar la síntesis aditiva del espectro en tiempo real o usar **`Exportar captura`** para guardar los archivos `.csv`, `.json` y `.wav`.

---

### Opción B: Modo En Vivo con HackRF One + GNU Radio — *2 terminales*

Para observar en directo con la antena corneta, el LNA/filtro SAWbird+ H1 y el HackRF One:

1. **Terminal 1 — Iniciar el flujo de GNU Radio (Productor ZMQ en puerto `5555`):**
   ```bash
   python gnuradio/receptor_hi_flowgraph.py
   ```
   *(Deja esta terminal abierta; se encarga de leer el HackRF por USB, aplicar la FFT de 1024 puntos, restar la calibración `cal_50ohm.npy` y publicar en `tcp://127.0.0.1:5555`).*

2. **Terminal 2 — Iniciar la Interfaz Gráfica (Consumidor ZMQ):**
   ```bash
   python -m src.ui.main_window
   ```
   Una vez abierta la ventana, haz clic en el botón **`En vivo`** de la barra inferior. El indicador superior cambiará a **`EN VIVO (HACKRF)`** y graficará la señal real de la antena.

> **Calibración de piso de ruido (50 Ohms):** Si necesitas regenerar el vector de referencia `cal_50ohm.npy` antes de observar, coloca la carga terminadora de 50 Ohms en la entrada del receptor y ejecuta:
> ```bash
> python gnuradio/receptor_hi_flowgraph.py --calibrate
> ```

---

## Descripción Técnica

El sistema reemplaza las etapas tradicionales de detección analógica posterior al acondicionamiento de radiofrecuencia (RF) mediante una arquitectura basada en Radio Definida por Software (SDR). El flujo de datos abarca desde la lectura de muestras complejas en cuadratura (I/Q) hasta la estimación de parámetros cinemáticos de nubes de gas galáctico mediante el Efecto Doppler no relativista:

`v_r = c * (f_0 - f) / f_0`

### Módulos Principales

1. **Adquisición y Procesamiento en Banda Base (SDR / DSP):**
   * Recepción de muestras digitales provenientes de un transceptor **HackRF One** (frecuencia central de 1420.40575 MHz, ancho de banda de 2 MHz) e integración con **GNU Radio**.
   * Cálculo de la Densidad Espectral de Potencia (PSD) mediante FFT de 1024 canales e integración temporal en potencia lineal para incrementar la Relación Señal a Ruido (SNR).
2. **Análisis Astrofísico y Sonificación:**
   * Sustracción automática de curvatura de línea base (*baseline* instrumental) mediante ajuste polinomial de segundo grado en los bordes de banda.
   * Conversión del dominio de frecuencias al dominio de velocidades radiales (`v_r`), estimación de flujo integrado y síntesis de audio en tiempo real (`sonification.py`).
3. **Visualización Científica e Interfaz de Usuario (`RADIOASTROLAB`):**
   * Despliegue simultáneo del espectro en frecuencia (MHz), espectro en velocidad radial (km/s) con regiones Doppler y espectrograma histórico (*waterfall*).
   * Conmutación en caliente entre hardware en vivo (`HackRFSource`), grabaciones reales (`.npz`) y simulador paramétrico interactivo.

## Arquitectura del Proyecto

```text
RadioTelescopio-Instituto-de-Astronomia/
├── gnuradio/
│   └── receptor_hi_flowgraph.py  # Flowgraph de GNU Radio + bloque DSP y servidor ZMQ PUB
├── data_samples/                 # Grabaciones reales (.npz) para reproducción y pruebas
├── src/
│   ├── contracts.py              # Contratos de datos (RawSpectrumBlock, AstroMetrics, UIFramePacket)
│   ├── acquisition/              # Fuentes de datos: HackRFSource (ZMQ) y MockSDRSource
│   ├── physics/                  # Conversión Doppler, limpieza de baseline y sonificación WAV
│   │   ├── hi_calculator.py
│   │   └── sonification.py
│   └── ui/                       # Interfaz gráfica PyQt6 + PyQtGraph
│       ├── main_window.py        # Ventana principal RADIOASTROLAB
│       └── qt_bridge.py          # Worker QThread y reproductor RecordedNPZSource
├── cal_50ohm.npy                 # Vector de calibración de respuesta en frecuencia (50 Ohms)
├── main.py                       # Script CLI de diagnóstico en consola
└── requirements.txt              # Dependencias de Python
```