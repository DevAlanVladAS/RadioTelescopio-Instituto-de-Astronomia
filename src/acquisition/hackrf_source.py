from datetime import datetime, timezone
import zmq
import numpy as np
from src.contracts import RawSpectrumBlock, HI_REST_FREQ_MHZ


class HackRFSource:
    """
    Receptor en vivo que escucha los vectores de potencia ya calibrados (50 Ohms)
    y promediados (SNR) provenientes de GNU Radio vía ZeroMQ.
    """

    def __init__(
        self,
        zmq_address: str = "tcp://127.0.0.1:5555",
        num_bins: int = 1024,
        center_freq_mhz: float = HI_REST_FREQ_MHZ,
        sample_rate_mhz: float = 2.0,
        integration_time_s: float = 5.0,
        timeout_ms: int = 5000,
    ):
        self.num_bins = num_bins
        self.center_freq_mhz = center_freq_mhz
        self.sample_rate_mhz = sample_rate_mhz
        self.integration_time_s = integration_time_s

        half_bw = self.sample_rate_mhz / 2.0
        self.frequencies_mhz = np.linspace(
            self.center_freq_mhz - half_bw,
            self.center_freq_mhz + half_bw,
            self.num_bins,
            dtype=np.float64,
        )

        # Configuramos el socket ZMQ SUB
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 5)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(zmq_address)

    def read_spectrum(self) -> RawSpectrumBlock:
        """Lee el último vector de 1024 float32 publicado por GNU Radio."""
        try:
            # 1. Lectura bloqueante inicial (espera hasta que haya al menos 1 paquete)
            parts = self.sock.recv_multipart()
            raw_bytes = parts[-1]

            # 2. Drenamos los paquetes viejos acumulados en cola para tener tiempo real
            while True:
                try:
                    parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    raw_bytes = parts[-1]
                except zmq.Again:
                    break
        except zmq.Again as exc:
            raise TimeoutError(
                "No se recibieron datos de GNU Radio en  el puerto tcp://127.0.0.1:5555. "
                "Verifica que gnuradio/receptor_hi_flowgraph.py esté corriendo."
            ) from exc

        # Si GNU Radio nos envía varios vectores pegados en un solo paquete ZMQ, tomamos los últimos 1024 floats
        floats = np.frombuffer(raw_bytes, dtype=np.float32)
        if len(floats) < self.num_bins:
            raise ValueError(f"Paquete incompleto recibido: {len(floats)} floats")

        power_db = floats[-self.num_bins:].astype(np.float64)

        return RawSpectrumBlock(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="EN_VIVO",
            center_freq_mhz=self.center_freq_mhz,
            sample_rate_mhz=self.sample_rate_mhz,
            integration_time_s=self.integration_time_s,
            frequencies_mhz=self.frequencies_mhz,
            power_db=power_db,
        )

    def close(self):
        self.sock.close()
        self.ctx.term()