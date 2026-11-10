#!/usr/bin/env python3
"""
Flujo DSP en GNU Radio para el Radiotelescopio HI (21 cm).
Cadena: HackRF One (Soapy) -> Stream to Vector (1024) -> FFT -> SNR & 50-Ohm Calibrator -> ZMQ PUB Sink
"""

import argparse
import signal
import sys
import time
import os
import numpy as np
from gnuradio import gr, blocks, fft, zeromq, soapy
from gnuradio.fft import window


class SNRAndCalibratorBlock(gr.sync_block):
    """
    Embedded Python Block de GNU Radio:
    1. Calcula densidad espectral de potencia |X[k]|^2.
    2. Elimina el pico DC central del HackRF One.
    3. Acumula/promedia espectros (IIR / exponencial) para mejorar la SNR.
    4. Guarda o resta la curva de calibracion de la carga de 50 Ohms.
    """

    def __init__(self, vec_len: int = 1024, alpha: float = 0.02, cal_file: str = "cal_50ohm.npy", save_cal: bool = False):
        gr.sync_block.__init__(
            self,
            name="SNR & 50-Ohm Calibrator",
            in_sig=[(np.complex64, vec_len)],
            out_sig=[(np.float32, vec_len)],
        )
        self.vec_len = vec_len
        self.alpha = alpha
        self.cal_file = cal_file
        self.save_cal = save_cal

        self.avg_power = np.zeros(vec_len, dtype=np.float64)
        self.initialized = False

        # Cargamos la curva de referencia de 50 Ohms si ya existe y no estamos en modo calibracion
        self.baseline_db = np.zeros(vec_len, dtype=np.float64)
        if not self.save_cal and os.path.exists(self.cal_file):
            self.baseline_db = np.load(self.cal_file)
            print(f"[CAL] Curva de 50 Ohms cargada desde '{self.cal_file}'. Aplicando sustracción de ruido interno.")
        elif self.save_cal:
            print(f"[CAL] MODO CALIBRACIÓN ACTIVO: Asegúrate de tener la resistencia de 50 Ohms conectada.")

    def work(self, input_items, output_items):
        in_vecs = input_items[0]
        out_vecs = output_items[0]

        for i in range(len(in_vecs)):
            spectrum = in_vecs[i]

            # 1. Magnitud al cuadrado normalizada
            power = (np.abs(spectrum) ** 2) / (self.vec_len ** 2)

            # 2. Suprimir el pico DC central del HackRF interpolando los bins vecinos
            mid = self.vec_len // 2
            power[mid - 2 : mid + 3] = 0.5 * (power[mid - 3] + power[mid + 3])

            # 3. Promediado temporal para elevar la SNR
            if not self.initialized:
                self.avg_power = power.astype(np.float64)
                self.initialized = True
            else:
                self.avg_power = (1.0 - self.alpha) * self.avg_power + self.alpha * power

            # 4. Convertir a dB y restar la referencia de 50 Ohms (si existe)
            power_db = 10.0 * np.log10(self.avg_power + 1e-12)
            calibrated_db = power_db - self.baseline_db

            out_vecs[i][:] = calibrated_db.astype(np.float32)

        return len(output_items[0])

    def save_current_baseline(self):
        """Guarda el espectro acumulado actual como referencia de 50 Ohms."""
        if self.initialized:
            ref_db = 10.0 * np.log10(self.avg_power + 1e-12)
            np.save(self.cal_file, ref_db)
            print(f"\n[OK] Referencia de 50 Ohms guardada exitosamente en '{self.cal_file}'.")


class ReceptorHIFlowgraph(gr.top_block):
    """Diagrama de bloques principal de GNU Radio."""

    def __init__(
        self,
        center_freq_hz: float = 1420.40575e6,
        samp_rate_hz: float = 2e6,
        fft_size: int = 1024,
        lna_gain: int = 32,
        vga_gain: int = 32,
        zmq_address: str = "tcp://127.0.0.1:5555",
        cal_file: str = "cal_50ohm.npy",
        save_cal: bool = False,
    ):
        super().__init__("Receptor HI 21cm - IA UNAM")

        self.fft_size = fft_size

        # 1. Bloque Fuente: Soapy HackRF Source (RF Amp APAGADO por seguridad)
        dev = "driver=hackrf"
        stream_args = ""
        tune_args = [""]
        settings = [""]
        self.sdr_source = soapy.source(dev, "fc32", 1, "", stream_args, tune_args, settings)
        self.sdr_source.set_sample_rate(0, samp_rate_hz)
        self.sdr_source.set_frequency(0, center_freq_hz)
        self.sdr_source.set_gain(0, "AMP", 0)       # ¡0 dB SIEMPRE para proteger el MGA-81563!
        self.sdr_source.set_gain(0, "LNA", lna_gain) # Ganancia IF segura (0-40 dB)
        self.sdr_source.set_gain(0, "VGA", vga_gain) # Ganancia Baseband segura (0-62 dB)

        # 2. Bloque Stream to Vector (paquetes de 1024 muestras)
        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)

        # 3. Bloque FFT (Forward, centrada con Shift y ventana Blackman-Harris)
        self.fft_block = fft.fft_vcc(
            fft_size,
            True,
            window.blackmanharris(fft_size),
            True,
            1,
        )

        # 4. Bloque Python embebido: Mejora de SNR + Calibración 50 Ohms
        self.snr_cal_block = SNRAndCalibratorBlock(
            vec_len=fft_size,
            alpha=0.01,
            cal_file=cal_file,
            save_cal=save_cal,
        )

        # 5. Bloque ZMQ PUB Sink (envía vectores de 1024 float32 al backend)
        self.zmq_sink = zeromq.pub_sink(
            gr.sizeof_float,
            fft_size,
            zmq_address,
            100,
            False,
            -1,
            "",
        )

        # Conectar la cadena de bloques en GNU Radio
        self.connect(
            (self.sdr_source, 0),
            (self.stream_to_vec, 0),
            (self.fft_block, 0),
            (self.snr_cal_block, 0),
            (self.zmq_sink, 0),
        )


def main():
    parser = argparse.ArgumentParser(description="Flowgraph GNU Radio para HackRF One")
    parser.add_argument("--freq", type=float, default=1420.40575e6, help="Frecuencia central en Hz")
    parser.add_argument("--rate", type=float, default=2e6, help="Sample rate en Hz")
    parser.add_argument("--calibrate", action="store_true", help="Guardar curva de ruido con carga de 50 Ohms (5 seg)")
    args = parser.parse_args()

    tb = ReceptorHIFlowgraph(
        center_freq_hz=args.freq,
        samp_rate_hz=args.rate,
        save_cal=args.calibrate,
    )

    tb.start()
    print(f"[GNU RADIO] Flowgraph corriendo en {args.freq/1e6:.5f} MHz -> Publicando en tcp://127.0.0.1:5555")

    if args.calibrate:
        print("[CAL] Promediando ruido térmico de 50 Ohms durante 5 segundos...")
        time.sleep(5.0)
        tb.snr_cal_block.save_current_baseline()
        tb.stop()
        tb.wait()
        sys.exit(0)

    def sig_handler(sig=None, frame=None):
        print("\n[GNU RADIO] Deteniendo recepción de forma segura...")
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    main()