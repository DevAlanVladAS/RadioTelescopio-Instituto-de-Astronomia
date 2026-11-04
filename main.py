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
        print("[INFO] Iniciando pipeline en MODO SIMULACIÓN...")
        source = MockSDRSource(num_bins=1024, sample_rate_mhz=2.0, integration_time_s=5.0)
    else:
        raise NotImplementedError(
            "El modo 'live' requiere enlazar HackRF/ZMQ en src/acquisition/hackrf_source.py"
        )
    for i in range(3):
        raw_block = source.read_spectrum()
        ui_packet = process_hi_spectrum(raw_block)
        resumen = ui_packet.to_dict()["metricas"]

        print(f"\n--- Frame #{i + 1} ({ui_packet.timestamp}) ---")
        print(f"Muestras espectrales: {len(ui_packet.frequencies_mhz)} puntos")
        print(f"Métricas calculadas : {json.dumps(resumen, indent=2)}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()