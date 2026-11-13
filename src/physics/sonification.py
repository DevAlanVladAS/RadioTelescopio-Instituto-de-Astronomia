import numpy as np
from scipy.io.wavfile import write
from src.contracts import UIFramePacket

SAMPLE_RATE = 44100  # Calidad estandar de audio (44.1 kHz)


def _spectrum_to_audio(packet: UIFramePacket, duration_s: float, smooth_edges: bool = True) -> np.ndarray:
    """
    Convierte los 1024 puntos limpios de UIFramePacket en una onda de audio.
    """
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    power = np.asarray(packet.power_db, dtype=float)
    noise_floor = np.median(power)
    above_noise = np.clip(power - noise_floor, 0.0, None)

    max_val = np.max(above_noise)
    if max_val > 0:
        amplitudes = (above_noise / max_val) ** 2
    else:
        amplitudes = np.zeros_like(above_noise)

    audible_freqs = np.linspace(220.0, 880.0, len(power))

    audio = np.zeros_like(t)
    active_bins = np.where(amplitudes > 0.08)[0]
    for idx in active_bins:
        audio += amplitudes[idx] * np.sin(2.0 * np.pi * audible_freqs[idx] * t)

    audio += 0.015 * np.random.normal(0, 1, n_samples)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.7 * (audio / peak)

    if smooth_edges and n_samples > 200:
        fade = np.linspace(0.0, 1.0, 100)
        audio[:100] *= fade
        audio[-100:] *= fade[::-1]

    return audio


def generate_realtime_chunk(packet: UIFramePacket, frame_duration_s: float = 0.1) -> np.ndarray:
    return _spectrum_to_audio(packet, frame_duration_s, smooth_edges=True).astype(np.float32)


def export_simulation_wav(packet: UIFramePacket, filename: str = "sonido_hidrogeno.wav", duration_s: float = 4.0):
    audio = _spectrum_to_audio(packet, duration_s, smooth_edges=False)
    audio_int16 = np.int16(audio * 32767)
    write(filename, SAMPLE_RATE, audio_int16)