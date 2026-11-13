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
    el paquete listo con el espectro corregido, velocidades radiales y métricas HI.
    """
    freqs = np.array(raw_block.frequencies_mhz, dtype=float)
    power = np.array(raw_block.power_db, dtype=float)
    n = len(freqs)

    # 1. Frecuencia -> velocidad radial (efecto Doppler no relativista)
    velocities = SPEED_OF_LIGHT_KMS * (HI_REST_FREQ_MHZ - freqs) / HI_REST_FREQ_MHZ

    # 2. Quitar la curvatura del fondo (baseline) usando el 15% de cada borde
    edge = max(1, int(n * 0.15))
    is_edge = np.zeros(n, dtype=bool)
    is_edge[:edge] = True
    is_edge[-edge:] = True

    coeffs = np.polyfit(freqs[is_edge], power[is_edge], deg=2)
    baseline = np.polyval(coeffs, freqs)

    clean = power - baseline
    noise_level = np.median(power[is_edge])
    corrected_power = clean + noise_level

    # 3. Buscar el pico de emisión (ignorando el 5% de los extremos)
    guard = max(1, int(n * 0.05))
    if n > 2 * guard:
        peak_idx = int(np.argmax(clean[guard:-guard])) + guard
    else:
        peak_idx = int(np.argmax(clean))

    peak_height = float(clean[peak_idx])
    peak_power_db = float(corrected_power[peak_idx])

    # 4. Relación señal/ruido (SNR)
    noise_std = float(np.std(clean[is_edge]))
    snr = max(0.0, peak_height) / max(noise_std, 1e-6)

    # Umbral físico: en 1024 puntos de puro ruido gaussiano, el pico aleatorio
    # siempre ronda ~3.0 - 3.5 sigma. Si SNR < 4.2, estamos viendo solo piso de ruido.
    if snr < 4.2:
        peak_freq = HI_REST_FREQ_MHZ
        peak_vel = 0.0
    else:
        peak_freq = float(freqs[peak_idx])
        peak_vel = float(velocities[peak_idx])

    # 5. Flujo integrado: área bajo la curva sobre el fondo
    above = np.clip(clean, 0.0, None)
    dv = abs(float(velocities[1] - velocities[0])) if n > 1 else 1.0
    integrate = getattr(np, "trapezoid", None) or np.trapz
    flux = float(integrate(above, dx=dv))

    # 6. Empaquetar en la estructura oficial del proyecto
    metrics = AstroMetrics(
        peak_radial_velocity_kms=round(peak_vel, 2),
        peak_frequency_mhz=round(peak_freq, 5),
        peak_power_db=round(peak_power_db, 2),
        integrated_flux=round(flux, 2),
        snr_db=round(snr, 2),
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