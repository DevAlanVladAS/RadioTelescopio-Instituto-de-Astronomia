from datetime import datetime, timezone
import numpy as np
from src.contracts import RawSpectrumBlock, HI_REST_FREQ_MHZ


class MockSDRSource:
    """
    Generador de señales sintéticas de la línea de 21 cm (HI) 
    para desarrollo.
    """
    def __init__(
        self,
        num_bins: int = 1024,
        sample_rate_mhz: float = 2.0,
        integration_time_s: float = 5.0
    ):
        self.num_bins = num_bins
        self.center_freq_mhz = HI_REST_FREQ_MHZ
        self.sample_rate_mhz = sample_rate_mhz
        self.integration_time_s = integration_time_s

        # Vector de frecuencias desde (f0 - BW/2) hasta (f0 + BW/2)
        half_bw = self.sample_rate_mhz / 2.0
        self.frequencies_mhz = np.linspace(
            self.center_freq_mhz - half_bw,
            self.center_freq_mhz + half_bw,
            self.num_bins,
            dtype=np.float64
        )

    def read_spectrum(self) -> RawSpectrumBlock:
        """Simula la entrega de un bloque espectral ya promediado."""
        # 1. Piso de ruido térmico alrededor de -85 dB
        noise_sigma = 0.8 / np.sqrt(self.integration_time_s)
        noise_db = np.random.normal(loc=-85.0, scale=noise_sigma, size=self.num_bins)

        # 2. Simular una nube de HI con corrimiento Doppler (+0.20 MHz -> acercándose a ~-42 km/s)
        shifted_center_mhz = self.center_freq_mhz + 0.20
        line_width_mhz = 0.08
        amplitude_db = 8.5  # Altura del pico sobre el piso de ruido

        hi_profile = amplitude_db * np.exp(
            -0.5 * ((self.frequencies_mhz - shifted_center_mhz) / line_width_mhz) ** 2
        )

        power_db = noise_db + hi_profile

        return RawSpectrumBlock(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="SIMULACION",
            center_freq_mhz=self.center_freq_mhz,
            sample_rate_mhz=self.sample_rate_mhz,
            integration_time_s=self.integration_time_s,
            frequencies_mhz=self.frequencies_mhz,
            power_db=power_db
        )