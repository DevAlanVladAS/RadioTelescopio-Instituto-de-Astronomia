"""
Worker en QThread para RADIOASTROLAB (PyQt).
Soporta:
- Modo 'live' (HackRF One + GNU Radio vía ZMQ).
- Modo 'simulation' (Reproducción de grabación real .npz si existe, o MockSDRSource sintético).
- Promediado dinámico en vivo con set_integration(segundos) para que el selector (1s, 5s, 15s)
  funcione tanto en directo como reproduciendo una grabación previa en la Noche de las Estrellas.
"""

import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

from src.contracts import RawSpectrumBlock, UIFramePacket, HI_REST_FREQ_MHZ
from src.acquisition.mock_source import MockSDRSource
from src.acquisition.hackrf_source import HackRFSource
from src.physics.hi_calculator import process_hi_spectrum


class RecordedNPZSource:
    """
    Reproduce en bucle continuo una grabación real previa (.npz) capturada con el HackRF.
    Ideal para demostraciones con alta interferencia externa (Noche de las Estrellas).
    """

    def __init__(self, npz_path: Path, integration_time_s: float = 5.0):
        data = np.load(npz_path)
        self.frequencies_mhz = data["frequencies_mhz"].astype(np.float64)
        raw_power = data["power_db"].astype(np.float64)
        # Soporta tanto un solo espectro (1D) como una grabación de N frames (2D: [N, 1024])
        self.frames = raw_power if raw_power.ndim == 2 else raw_power[np.newaxis, :]
        self.center_freq_mhz = float(data.get("center_freq_mhz", HI_REST_FREQ_MHZ))
        self.sample_rate_mhz = float(data.get("sample_rate_mhz", 2.0))
        self.integration_time_s = integration_time_s
        self.idx = 0
        self.rng = np.random.default_rng()

    def read_spectrum(self) -> RawSpectrumBlock:
        base_frame = self.frames[self.idx % len(self.frames)].copy()
        self.idx += 1

        # Si el archivo tiene un solo espectro base, añadimos micro-variación térmica realista
        if len(self.frames) == 1:
            base_frame += self.rng.normal(0.0, 0.35, size=base_frame.shape)

        return RawSpectrumBlock(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="SIMULACION",
            center_freq_mhz=self.center_freq_mhz,
            sample_rate_mhz=self.sample_rate_mhz,
            integration_time_s=self.integration_time_s,
            frequencies_mhz=self.frequencies_mhz,
            power_db=base_frame,
        )

    def close(self) -> None:
        pass


class SpectrumAcquisitionWorker(QThread):
    """Hilo de adquisición conectado a la interfaz PyQt de Katia."""

    packet = pyqtSignal(dict)       # Emite el diccionario del contrato UIFramePacket
    state = pyqtSignal(str, str)    # Claves: CONECTANDO | EN_VIVO | INTEGRANDO | ERROR

    def __init__(self, mode: str = "simulation", recording_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.mode = mode  # "simulation" o "live"
        self.integration_time_s: float = 5.0
        self._running: bool = False
        self._paused: bool = False
        self._integration_changed: bool = False
        self.last_packet: Optional[UIFramePacket] = None

        default_rec = Path(__file__).resolve().parents[2] / "data_samples" / "grabacion_noche_estrellas.npz"
        self.recording_path = Path(recording_path) if recording_path else default_rec

    def set_integration(self, segundos: float) -> None:
        """Conectar al selector 'Integración' (ej. 1, 5, 15). Modifica el buffer de promediado en vivo."""
        self.integration_time_s = max(0.5, float(segundos))
        self._integration_changed = True
        if self.isRunning() and not self._paused:
            self.state.emit(
                "INTEGRANDO",
                f"Ajustando ventana de integración a {self.integration_time_s:.0f} s...",
            )

    def set_mode(self, mode: str) -> None:
        """Conectar a los botones 'En vivo' ('live') y 'Simulación' ('simulation')."""
        nuevo_modo = "live" if mode.lower() in ("live", "en_vivo", "en vivo") else "simulation"
        if nuevo_modo != self.mode:
            was_running = self.isRunning()
            if was_running:
                self.stop()
            self.mode = nuevo_modo
            if was_running:
                self.start()

    def pause(self, paused: Optional[bool] = None) -> None:
        """Conectar al botón 'Pausar' de la barra inferior."""
        self._paused = (not self._paused) if paused is None else bool(paused)

    def stop(self) -> None:
        """Conectar al botón 'Detener'."""
        self._running = False
        self._paused = False
        self.wait(2000)

    def run(self) -> None:
        self._running = True
        self._paused = False
        source = None
        history_buffer: deque = deque(maxlen=300)

        try:
            if self.mode == "live":
                self.state.emit("CONECTANDO", "Conectando a GNU Radio (tcp://127.0.0.1:5555)...")
                source = HackRFSource(
                    num_bins=1024,
                    sample_rate_mhz=2.0,
                    integration_time_s=self.integration_time_s,
                )
                modo_label = "HackRF One (En Vivo)"
            else:
                if self.recording_path.exists():
                    self.state.emit("CONECTANDO", f"Cargando grabación previa ({self.recording_path.name})...")
                    source = RecordedNPZSource(self.recording_path, integration_time_s=self.integration_time_s)
                    modo_label = "Grabación HI Previa"
                else:
                    self.state.emit("CONECTANDO", "Inicializando modelo sintético de hidrógeno neutro (21 cm)...")
                    source = MockSDRSource(
                        num_bins=1024,
                        sample_rate_mhz=2.0,
                        integration_time_s=1.0,  # Base de 1s para que el buffer del worker controle el suavizado
                    )
                    modo_label = "Simulación HI"

            while self._running:
                if self._paused:
                    time.sleep(0.1)
                    continue

                raw_block = source.read_spectrum()
                raw_block.integration_time_s = self.integration_time_s

                # Buffer de integración real (~10 FPS: 1s = 10 frames, 5s = 50 frames, 15s = 150 frames)
                target_frames = max(1, int(self.integration_time_s * 10))
                history_buffer.append(raw_block.power_db.copy())
                while len(history_buffer) > target_frames:
                    history_buffer.popleft()

                # Promediado en potencia lineal -> dB
                lin_stack = [10.0 ** (p / 10.0) for p in history_buffer]
                raw_block.power_db = 10.0 * np.log10(np.mean(lin_stack, axis=0) + 1e-12)

                ui_frame = process_hi_spectrum(raw_block)
                self.last_packet = ui_frame

                if self._integration_changed or len(history_buffer) == 1:
                    self._integration_changed = False
                    self.state.emit(
                        "EN_VIVO",
                        f"Activo: {modo_label} | Integración: {self.integration_time_s:.0f} s",
                    )

                payload = ui_frame.to_dict()
                payload["frequencies_mhz"] = payload["frecuencias_mhz"]
                payload["radial_velocities_kms"] = payload["velocidades_radiales_kms"]
                payload["power_db"] = payload["potencia_db"]
                payload["metrics"] = payload["metricas_hi"]

                self.packet.emit(payload)
                time.sleep(0.1)

        except Exception as exc:
            self.state.emit("ERROR", str(exc))
        finally:
            if source is not None and hasattr(source, "close"):
                source.close()
            self._running = False

    def export_last_frame(self, base_filepath: str) -> tuple[str, str]:
        """Conectar al botón 'Exportar captura'."""
        if self.last_packet is None:
            raise RuntimeError("No hay ningún espectro en memoria para exportar.")

        base = Path(base_filepath).with_suffix("")
        csv_path = f"{base}.csv"
        json_path = f"{base}.json"

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["frecuencia_mhz", "velocidad_radial_kms", "potencia_db"])
            for freq, vel, pwr in zip(
                self.last_packet.frequencies_mhz,
                self.last_packet.radial_velocities_kms,
                self.last_packet.power_db,
            ):
                writer.writerow([f"{freq:.6f}", f"{vel:.4f}", f"{pwr:.4f}"])

        with open(json_path, mode="w", encoding="utf-8") as f_json:
            json.dump(self.last_packet.to_dict(), f_json, indent=2, ensure_ascii=False)

        return csv_path, json_path