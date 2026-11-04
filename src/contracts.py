from dataclasses import dataclass, asdict
from typing import Literal
import numpy as np
from numpy.typing import NDArray

# Constantes fisicas fundamentales para el proyecto
HI_REST_FREQ_MHZ: float = 1420.40575177  # Frecuencia en reposo de la linea de 21 cm (MHz)
SPEED_OF_LIGHT_KMS: float = 299792.458   # Velocidad de la luz en el vacio (km/s)


@dataclass
class RawSpectrumBlock:
    """
    Paquete que entrega el modulo de Adquisicion (Alan)
    hacia el modulo de Fisica (Ricardo).
    """
    timestamp: str
    mode: Literal["EN_VIVO", "SIMULACION"]
    center_freq_mhz: float
    sample_rate_mhz: float
    integration_time_s: float
    frequencies_mhz: NDArray[np.float64]  # Vector de frecuencias (Longitud N)
    power_db: NDArray[np.float64]         # Espectro de potencia promediado en dB (Longitud N)


@dataclass
class AstroMetrics:
    """
    Metricas escalares calculadas por el modulo de Fisica (Ricardo)
    para mostrar en las tarjetas informativas de la UI.
    """
    peak_radial_velocity_kms: float
    peak_frequency_mhz: float
    integrated_flux: float
    snr_db: float


@dataclass
class UIFramePacket:
    """
    Contrato final de datos que recibe la Interfaz de Usuario (Katia)
    en cada actualizacion de pantalla.
    """
    timestamp: str
    mode: Literal["EN_VIVO", "SIMULACION"]
    center_freq_mhz: float
    integration_time_s: float
    
    # Ejes para graficar (Longitud N, ej. N = 1024)
    frequencies_mhz: NDArray[np.float64]
    velocities_kms: NDArray[np.float64]
    power_db: NDArray[np.float64]
    
    # Indicadores numericos
    metrics: AstroMetrics

    def to_dict(self) -> dict:
        """Convierte el paquete a un diccionario serializable (util para JSON / WebSockets)."""
        return {
            "timestamp": self.timestamp,
            "modo": self.mode,
            "frecuencia_central_mhz": self.center_freq_mhz,
            "tiempo_integracion_s": self.integration_time_s,
            "eje_frecuencias_mhz": self.frequencies_mhz.tolist(),
            "eje_velocidades_kms": self.velocities_kms.tolist(),
            "espectro_potencia_db": self.power_db.tolist(),
            "metricas": {
                "velocidad_radial_pico_kms": round(self.metrics.peak_radial_velocity_kms, 2),
                "frecuencia_pico_mhz": round(self.metrics.peak_frequency_mhz, 5),
                "flujo_integrado": round(self.metrics.integrated_flux, 2),
                "snr_db": round(self.metrics.snr_db, 2),
            }
        }