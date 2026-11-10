import argparse
import time
import json
from src.acquisition.mock_source import MockSDRSource
from src.physics.hi_calculator import process_hi_spectrum


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Adquisición y Procesamiento para Radiotelescopio HI"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "live"],
        default="simulation",
        help="Modo de operación: 'simulation' (datos sintéticos) o 'live' (HackRF One)",
    )
    args = parser.parse_args()

    if args.mode == "simulation":
        print("[INFO] Iniciando pipeline en MODO SIMULACIÓN")
        source = MockSDRSource(num_bins=1024, sample_rate_mhz=2.0, integration_time_s=5.0)
    else:
        from src.acquisition.hackrf_source import HackRFSource
        print("pipeline en MODO EN VIVO (ZMQ <- GNU Radio)...")
        source = HackRFSource(num_bins=1024, sample_rate_mhz=2.0, integration_time_s=5.0)

    print("Streaming de datos en curso")
    try:
        while True:
            raw_block = source.read_spectrum()
            ui_packet = process_hi_spectrum(raw_block)
            print(f"[{ui_packet.timestamp}] Pico Vel: {ui_packet.metrics.peak_radial_velocity_kms} km/s | SNR: {ui_packet.metrics.snr_db} dB")
    except KeyboardInterrupt:
        print("\n[INFO] Transmision detenida.")
        source.close()

if __name__ == "__main__":
    main()