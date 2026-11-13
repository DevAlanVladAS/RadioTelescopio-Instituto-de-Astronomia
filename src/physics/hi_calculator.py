import numpy as np
from src.contracts import (
    RawSpectrumBlock,
    AstroMetrics,
    UIFramePacket,
    HI_REST_FREQ_MHZ,
    SPEED_OF_LIGHT_KMS,
)


def process_hi_spectrum(raw_block: RawSpectrumBlock) -> UIFramePacket:
    """
    Recibe un bloque de espectro (frecuencias y potencia en dB) y devuelve
    el paquete listo con:
      - el espectro corregido,
      - la velocidad radial de cada punto,
      - métricas de la línea de hidrógeno (pico, flujo, SNR).
    """
    freqs = np.array(raw_block.frequencies_mhz, dtype=float)
    power = np.array(raw_block.power_db, dtype=float)
    n = len(freqs)

    # 1. Frecuencia -> velocidad radial (efecto Doppler no relativista)
    #    v = c * (f0 - f) / f0
    #    Si f < f0 la fuente se aleja (v > 0). Si f > f0 se acerca (v < 0).
    velocities = SPEED_OF_LIGHT_KMS * (HI_REST_FREQ_MHZ - freqs) / HI_REST_FREQ_MHZ

    # 2. Quitar la "curvatura" del fondo (baseline)
    #    El SDR no responde igual en todas las frecuencias, así que el fondo
    #    no es plano. Lo ajustamos con una parábola usando solo los bordes
    #    del espectro (15% de cada lado), donde suponemos que no hay señal de HI.
    edge = max(1, int(n * 0.15))
    is_edge = np.zeros(n, dtype=bool)
    is_edge[:edge] = True
    is_edge[-edge:] = True

    coeffs = np.polyfit(freqs[is_edge], power[is_edge], deg=2)  # ajuste a x^2
    baseline = np.polyval(coeffs, freqs)                        # parábola evaluada

    clean = power - baseline   # señal con fondo en ~0 dB

    # Para graficar, se le devuelve el nivel típico del ruido (mediana de los bordes)
    noise_level = np.median(power[is_edge])
    corrected_power = clean + noise_level

    # 3. Buscar el pico de emisión (el máximo de la señal limpia)
    #    Se ignora el 5% de cada extremo porque ahí el filtro cae abruptamente
    #    y puede dar falsos máximos.
    guard = max(1, int(n * 0.05))
    if n > 2 * guard:
        peak_idx = np.argmax(clean[guard:-guard]) + guard
    else:
        peak_idx = np.argmax(clean)

    peak_freq = freqs[peak_idx]
    peak_vel = velocities[peak_idx]
    peak_height = clean[peak_idx]

    # 4. Relación señal/ruido
    #    Altura del pico dividida entre la desviación estándar del ruido.
    noise_std = np.std(clean[is_edge])
    snr = max(0.0, peak_height) / max(noise_std, 1e-6)

    # 5. Flujo integrado: área bajo la curva (regla del trapecio)
    #    Solo cuenta lo que está por encima del fondo (valores > 0).
    above = np.clip(clean, 0.0, None)
    dv = abs(velocities[1] - velocities[0]) if n > 1 else 1.0
    integrate = getattr(np, "trapezoid", None) or np.trapz   # compatibilidad NumPy 1.x / 2.x
    flux = integrate(above, dx=dv)

    # 6. Empaquetar en la estructura oficial del proyecto
    metrics = AstroMetrics(
        peak_radial_velocity_kms=round(float(peak_vel), 2),
        peak_frequency_mhz=round(float(peak_freq), 5),
        integrated_flux=round(float(flux), 2),
        snr_db=round(float(snr), 2),
    )

    return UIFramePacket(
        timestamp=raw_block.timestamp,
        mode=raw_block.mode,
        center_freq_mhz=raw_block.center_freq_mhz,
        integration_time_s=raw_block.integration_time_s,
        frequencies_mhz=freqs,
        velocities_kms=velocities,
        power_db=corrected_power,
        metrics=metrics,
    )
