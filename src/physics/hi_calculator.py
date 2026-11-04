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
    Transforma el espectro de frecuencias del SDR en velocidades radiales
    y calcula las métricas astrofísicas de la línea HI.
    """
    freqs = raw_block.frequencies_mhz
    power = raw_block.power_db

    # 1. Efecto Doppler: v_r = c * (f0 - f) / f0
    velocities_kms = SPEED_OF_LIGHT_KMS * (HI_REST_FREQ_MHZ - freqs) / HI_REST_FREQ_MHZ

    # 2. Detección del pico de emisión
    peak_idx = int(np.argmax(power))
    peak_freq_mhz = float(freqs[peak_idx])
    peak_vel_kms = float(velocities_kms[peak_idx])

    # 3. Estimación básica de SNR y flujo sobre el nivel base (baseline)
    baseline_db = float(np.median(power))
    noise_std_db = float(np.std(power[:100]))  # Desviación estándar en los bordes sin señal
    snr_db = float((power[peak_idx] - baseline_db) / max(noise_std_db, 1e-6))

    # Integración numérica del perfil sobre el baseline
    signal_above_base = np.clip(power - baseline_db, a_min=0.0, a_max=None)
    dv = abs(float(velocities_kms[1] - velocities_kms[0]))
    integrated_flux = float(np.trapezoid(signal_above_base, dx=dv))

    metrics = AstroMetrics(
        peak_radial_velocity_kms=peak_vel_kms,
        peak_frequency_mhz=peak_freq_mhz,
        integrated_flux=integrated_flux,
        snr_db=snr_db,
    )

    return UIFramePacket(
        timestamp=raw_block.timestamp,
        mode=raw_block.mode,
        center_freq_mhz=raw_block.center_freq_mhz,
        integration_time_s=raw_block.integration_time_s,
        frequencies_mhz=freqs,
        velocities_kms=velocities_kms,
        power_db=power,
        metrics=metrics,
    )