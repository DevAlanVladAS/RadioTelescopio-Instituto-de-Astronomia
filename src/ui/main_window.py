import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFrame,
    QFileDialog,
    QMessageBox,
    QSlider,
)
from src.contracts import HI_REST_FREQ_MHZ, SPEED_OF_LIGHT_KMS, RawSpectrumBlock
from src.physics.hi_calculator import process_hi_spectrum
from src.physics.sonification import SAMPLE_RATE, generate_realtime_chunk, export_simulation_wav
from src.ui.qt_bridge import RecordedNPZSource, SpectrumAcquisitionWorker


class ConfigurableSimulationSource:
    PRESETS = {
        "Grabación Noche de las Estrellas (.npz)": None,
        "Nube acercándose (−43.5 km/s)": {"vel_kms": -43.5, "amp_db": 9.5, "sigma_mhz": 0.032, "noise_db": 0.35},
        "Brazo espiral alejándose (+68.0 km/s)": {"vel_kms": 68.0, "amp_db": 7.5, "sigma_mhz": 0.045, "noise_db": 0.40},
        "Hidrógeno local en reposo (0.0 km/s)": {"vel_kms": 0.0, "amp_db": 11.0, "sigma_mhz": 0.025, "noise_db": 0.30},
        "Señal débil / Alto ruido (+22.0 km/s)": {"vel_kms": 22.0, "amp_db": 3.2, "sigma_mhz": 0.038, "noise_db": 0.85},
    }

    def __init__(self, npz_source=None, integration_time_s: float = 5.0):
        self.npz_source = npz_source
        self.integration_time_s = integration_time_s
        self.use_npz = npz_source is not None
        self.vel_kms = -43.5
        self.amp_db = 9.5
        self.sigma_mhz = 0.032
        self.noise_db = 0.35
        self.freqs = np.linspace(HI_REST_FREQ_MHZ - 1.0, HI_REST_FREQ_MHZ + 1.0, 1024)

    def apply_preset(self, name: str):
        cfg = self.PRESETS.get(name)
        if cfg is None and self.npz_source is not None:
            self.use_npz = True
            return
        self.use_npz = False
        if cfg:
            self.vel_kms = cfg["vel_kms"]
            self.amp_db = cfg["amp_db"]
            self.sigma_mhz = cfg["sigma_mhz"]
            self.noise_db = cfg["noise_db"]

    def read_spectrum(self) -> RawSpectrumBlock:
        if self.use_npz and self.npz_source is not None:
            return self.npz_source.read_spectrum()

        sdr_dome = -85.0 - 4.5 * ((self.freqs - HI_REST_FREQ_MHZ) ** 2)
        eff_noise = self.noise_db / np.sqrt(max(1.0, self.integration_time_s / 5.0))
        noise = np.random.normal(0.0, eff_noise, len(self.freqs))

        f_center = HI_REST_FREQ_MHZ * (1.0 - self.vel_kms / SPEED_OF_LIGHT_KMS)
        hi_signal = self.amp_db * np.exp(-0.5 * ((self.freqs - f_center) / self.sigma_mhz) ** 2)

        return RawSpectrumBlock(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="SIMULACION",
            center_freq_mhz=HI_REST_FREQ_MHZ,
            sample_rate_mhz=2.0,
            integration_time_s=self.integration_time_s,
            frequencies_mhz=self.freqs,
            power_db=sdr_dome + noise + hi_signal,
        )


class RadioTelescopioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RADIOASTROLAB — Observación de hidrógeno neutro – 21 cm")
        self.resize(1280, 690)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #060b19; color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
            QFrame#Card { background-color: #0d152a; border: 1px solid #1e293b; border-radius: 8px; padding: 6px 10px; }
            QPushButton { background-color: #131c31; border: 1px solid #26334d; border-radius: 5px; padding: 5px 12px; font-size: 11px; font-weight: 600; color: #cbd5e1; }
            QPushButton:hover { background-color: #1e293b; }
            QPushButton#StartBtn { background-color: #10b981; border: none; color: #022c22; font-weight: bold; }
            QPushButton#StartBtn:hover { background-color: #34d399; }
            QPushButton#ActiveMode { background-color: #2563eb; border: 1px solid #60a5fa; color: #ffffff; }
            QComboBox { background-color: #131c31; border: 1px solid #26334d; border-radius: 5px; padding: 4px 8px; font-size: 11px; color: #f8fafc; }
        """)

        self.waterfall_rows = 80
        self.waterfall_data = np.full((self.waterfall_rows, 1024), -90.0, dtype=np.float32)
        self.integration_s = 5.0
        self.paused = False

        self.audio_enabled = False
        self.audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        self.audio_stream.start()

        rec_path = Path(__file__).resolve().parents[2] / "data_samples" / "grabacion_noche_estrellas.npz"
        npz_src = RecordedNPZSource(rec_path, integration_time_s=self.integration_s) if rec_path.exists() else None
        self.source = ConfigurableSimulationSource(npz_source=npz_src, integration_time_s=self.integration_s)
        self.export_helper = SpectrumAcquisitionWorker(mode="simulation")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)

        # ================= ENCABEZADO COMPACTO + COORDENADAS =================
        header = QHBoxLayout()
        header.setSpacing(10)

        lbl_title = QLabel("RADIOASTROLAB")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 800; letter-spacing: 1px; color: #ffffff;")
        lbl_sub = QLabel("HI – 21 cm")
        lbl_sub.setStyleSheet("font-size: 11px; color: #f59e0b; font-weight: bold;")

        header.addWidget(lbl_title)
        header.addWidget(lbl_sub)
        header.addSpacing(12)

        # Coordenadas e integración integradas en cápsulas compactas
        header.addWidget(self._make_inline_pill("AR:", "18h 36m 00s"))
        header.addWidget(self._make_inline_pill("DEC:", "−08° 30′ 00″"))
        self.pill_int = self._make_inline_pill("Integración:", "5.0 s")
        header.addWidget(self.pill_int)

        header.addStretch()

        self.badge_mode = QLabel("SIMULACIÓN")
        self.badge_mode.setStyleSheet(
            "border: 1px solid #d97706; color: #fbbf24; background-color: #1c1917; "
            "padding: 3px 10px; border-radius: 5px; font-size: 10px; font-weight: bold;"
        )
        self.badge_state = QLabel("● Detenido")
        self.badge_state.setStyleSheet(
            "border: 1px solid #334155; color: #94a3b8; background-color: #0f172a; "
            "padding: 3px 10px; border-radius: 10px; font-size: 10px; font-weight: bold;"
        )
        header.addWidget(self.badge_mode)
        header.addWidget(self.badge_state)
        root.addLayout(header)

        # ================= FILA PRINCIPAL: DOS GRÁFICAS GRANDES LADO A LADO =================
        pg.setConfigOptions(antialias=True)
        row_plots = QHBoxLayout()
        row_plots.setSpacing(8)

        # --- Gráfica Izquierda: ESPECTRO DE FRECUENCIA ---
        freq_card = QFrame()
        freq_card.setObjectName("Card")
        freq_lay = QVBoxLayout(freq_card)
        freq_lay.setContentsMargins(6, 4, 6, 4)
        freq_lay.setSpacing(2)

        lbl_f_title = QLabel("ESPECTRO DE FRECUENCIA")
        lbl_f_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #94a3b8;")
        freq_lay.addWidget(lbl_f_title)

        self.freq_plot = pg.PlotWidget()
        self.freq_plot.setBackground("#0d152a")
        self.freq_plot.showGrid(x=True, y=True, alpha=0.18)
        self.freq_plot.setLabel("left", "Potencia (dB)")
        self.freq_plot.setLabel("bottom", "Frecuencia (MHz)")
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen("#f59e0b", width=2))
        self.freq_dot = self.freq_plot.plot(
            pen=None, symbol="o", symbolSize=8, symbolBrush="#fbbf24", symbolPen="w"
        )
        self.freq_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#38bdf8", style=Qt.PenStyle.DashLine))
        self.freq_plot.addItem(self.freq_vline)
        self.freq_tag = pg.TextItem(color="#ffffff", anchor=(1.05, 1.1))
        self.freq_plot.addItem(self.freq_tag)
        freq_lay.addWidget(self.freq_plot)
        row_plots.addWidget(freq_card)

        # --- Gráfica Derecha: ESPECTRO DE VELOCIDAD RADIAL ---
        vel_card = QFrame()
        vel_card.setObjectName("Card")
        vel_lay = QVBoxLayout(vel_card)
        vel_lay.setContentsMargins(6, 4, 6, 4)
        vel_lay.setSpacing(2)

        vel_hdr = QHBoxLayout()
        lbl_v_title = QLabel("ESPECTRO DE VELOCIDAD RADIAL")
        lbl_v_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #94a3b8;")
        lbl_blue = QLabel("◄ Acercamiento")
        lbl_blue.setStyleSheet("font-size: 10px; color: #38bdf8; font-weight: bold;")
        lbl_red = QLabel("Alejamiento ►")
        lbl_red.setStyleSheet("font-size: 10px; color: #f87171; font-weight: bold;")
        vel_hdr.addWidget(lbl_v_title)
        vel_hdr.addStretch()
        vel_hdr.addWidget(lbl_blue)
        vel_hdr.addSpacing(10)
        vel_hdr.addWidget(lbl_red)
        vel_lay.addLayout(vel_hdr)

        self.vel_plot = pg.PlotWidget()
        self.vel_plot.setBackground("#0d152a")
        self.vel_plot.showGrid(x=True, y=True, alpha=0.18)
        self.vel_plot.setLabel("left", "Potencia (dB)")
        self.vel_plot.setLabel("bottom", "Velocidad radial (km/s)")

        blue_region = pg.LinearRegionItem(values=[-300, 0], brush=(14, 165, 233, 22), movable=False)
        red_region = pg.LinearRegionItem(values=[0, 300], brush=(239, 68, 68, 22), movable=False)
        self.vel_plot.addItem(blue_region)
        self.vel_plot.addItem(red_region)

        self.vel_curve = self.vel_plot.plot(pen=pg.mkPen("#f59e0b", width=2))
        self.vel_dot = self.vel_plot.plot(
            pen=None, symbol="o", symbolSize=8, symbolBrush="#fbbf24", symbolPen="w"
        )
        self.vel_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#38bdf8", style=Qt.PenStyle.DashLine))
        self.vel_plot.addItem(self.vel_vline)
        self.vel_tag = pg.TextItem(color="#ffffff", anchor=(1.05, 1.1))
        self.vel_plot.addItem(self.vel_tag)
        vel_lay.addWidget(self.vel_plot)
        row_plots.addWidget(vel_card)

        # Mayor peso vertical (stretch=5) para que las gráficas dominen la ventana
        root.addLayout(row_plots, stretch=5)

        # ================= FILA DE MÉTRICAS COMPACTA =================
        row_metrics = QHBoxLayout()
        row_metrics.setSpacing(8)

        # Tarjeta 1: Velocidad Radial
        c_vel = QFrame()
        c_vel.setObjectName("Card")
        l_vel = QHBoxLayout(c_vel)
        l_vel.setContentsMargins(10, 4, 10, 4)
        t1 = QLabel("VELOCIDAD RADIAL:")
        t1.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: bold;")
        self.lbl_big_vel = QLabel("--")
        self.lbl_big_vel.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff;")
        u1 = QLabel("km/s")
        u1.setStyleSheet("font-size: 11px; color: #94a3b8;")
        self.pill_dir = QLabel("◄ ACERCAMIENTO")
        self.pill_dir.setStyleSheet(
            "border: 1px solid #0284c7; color: #38bdf8; background-color: #0c243c; "
            "border-radius: 5px; padding: 2px 8px; font-size: 10px; font-weight: bold;"
        )
        l_vel.addWidget(t1)
        l_vel.addWidget(self.lbl_big_vel)
        l_vel.addWidget(u1)
        l_vel.addStretch()
        l_vel.addWidget(self.pill_dir)
        row_metrics.addWidget(c_vel)

        # Tarjeta 2: Frecuencia Detectada
        c_freq = QFrame()
        c_freq.setObjectName("Card")
        l_freq = QHBoxLayout(c_freq)
        l_freq.setContentsMargins(10, 4, 10, 4)
        t2 = QLabel("FRECUENCIA PICO:")
        t2.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: bold;")
        self.lbl_big_freq = QLabel("--")
        self.lbl_big_freq.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff;")
        u2 = QLabel("MHz")
        u2.setStyleSheet("font-size: 11px; color: #94a3b8;")
        sub2 = QLabel(f"(f₀: {HI_REST_FREQ_MHZ:.3f})")
        sub2.setStyleSheet("font-size: 10px; color: #64748b;")
        l_freq.addWidget(t2)
        l_freq.addWidget(self.lbl_big_freq)
        l_freq.addWidget(u2)
        l_freq.addStretch()
        l_freq.addWidget(sub2)
        row_metrics.addWidget(c_freq)

        # Tarjeta 3: Potencia Espectral del Pico + SNR + Flujo
        c_pwr = QFrame()
        c_pwr.setObjectName("Card")
        l_pwr = QHBoxLayout(c_pwr)
        l_pwr.setContentsMargins(10, 4, 10, 4)
        t3 = QLabel("PICO:")
        t3.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: bold;")
        self.lbl_big_pwr = QLabel("--")
        self.lbl_big_pwr.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff;")
        u3 = QLabel("dB")
        u3.setStyleSheet("font-size: 11px; color: #94a3b8;")
        self.lbl_sub_snr = QLabel("SNR: -- dB | Flujo: --")
        self.lbl_sub_snr.setStyleSheet("font-size: 10px; color: #94a3b8;")
        l_pwr.addWidget(t3)
        l_pwr.addWidget(self.lbl_big_pwr)
        l_pwr.addWidget(u3)
        l_pwr.addStretch()
        l_pwr.addWidget(self.lbl_sub_snr)
        row_metrics.addWidget(c_pwr)

        root.addLayout(row_metrics, stretch=0)

        # ================= CASCADA · HISTORIAL DEL ESPECTRO (AMPLIADA) =================
        wf_card = QFrame()
        wf_card.setObjectName("Card")
        wf_lay = QVBoxLayout(wf_card)
        wf_lay.setContentsMargins(6, 4, 6, 4)
        wf_lay.setSpacing(2)

        lbl_wf = QLabel("CASCADA · HISTORIAL DEL ESPECTRO")
        lbl_wf.setStyleSheet("font-size: 10px; font-weight: bold; color: #94a3b8;")
        wf_lay.addWidget(lbl_wf)

        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setBackground("#0d152a")
        self.wf_plot.setLabel("left", "Tiempo")
        self.wf_plot.setLabel("bottom", "Frecuencia (MHz)")
        self.wf_plot.setMinimumHeight(140)
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        self.wf_img.setColorMap(pg.colormap.get("inferno"))
        wf_lay.addWidget(self.wf_plot)

        # Stretch=4 para que la cascada crezca proporcionalmente con la pantalla
        root.addWidget(wf_card, stretch=4)

        # ================= PANEL DE PARÁMETROS DE SIMULACIÓN =================
        sim_card = QFrame()
        sim_card.setObjectName("Card")
        sim_lay = QHBoxLayout(sim_card)
        sim_lay.setContentsMargins(10, 4, 10, 4)
        sim_lay.setSpacing(12)

        lbl_esc = QLabel("Escenario:")
        lbl_esc.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(ConfigurableSimulationSource.PRESETS.keys()))
        self.cmb_preset.currentTextChanged.connect(self.on_preset_changed)

        self.lbl_sld_vel = QLabel("Velocidad: -43.5 km/s")
        self.lbl_sld_vel.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: bold;")
        self.sld_vel = QSlider(Qt.Orientation.Horizontal)
        self.sld_vel.setRange(-150, 150)
        self.sld_vel.setValue(-44)
        self.sld_vel.valueChanged.connect(self.on_slider_changed)

        self.lbl_sld_amp = QLabel("Pico HI: 9.5 dB")
        self.lbl_sld_amp.setStyleSheet("font-size: 11px; color: #fbbf24; font-weight: bold;")
        self.sld_amp = QSlider(Qt.Orientation.Horizontal)
        self.sld_amp.setRange(1, 20)
        self.sld_amp.setValue(10)
        self.sld_amp.valueChanged.connect(self.on_slider_changed)

        self.lbl_sld_noise = QLabel("Ruido: 0.35 dB")
        self.lbl_sld_noise.setStyleSheet("font-size: 11px; color: #f87171; font-weight: bold;")
        self.sld_noise = QSlider(Qt.Orientation.Horizontal)
        self.sld_noise.setRange(1, 15)
        self.sld_noise.setValue(4)
        self.sld_noise.valueChanged.connect(self.on_slider_changed)

        for w in [
            lbl_esc, self.cmb_preset,
            self.lbl_sld_vel, self.sld_vel,
            self.lbl_sld_amp, self.sld_amp,
            self.lbl_sld_noise, self.sld_noise,
        ]:
            sim_lay.addWidget(w)

        root.addWidget(sim_card, stretch=0)

        # ================= CONTROLES INFERIORES =================
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.btn_start = QPushButton("▶ Iniciar")
        self.btn_start.setObjectName("StartBtn")
        self.btn_start.clicked.connect(self.start_acquisition)

        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("■ Detener")
        self.btn_stop.clicked.connect(self.stop_acquisition)

        lbl_int = QLabel("Integración:")
        lbl_int.setStyleSheet("color: #94a3b8; font-size: 11px; margin-left: 6px;")
        self.cmb_int = QComboBox()
        self.cmb_int.addItems(["1 s", "5 s", "15 s"])
        self.cmb_int.setCurrentIndex(1)
        self.cmb_int.currentIndexChanged.connect(self.change_integration)

        self.btn_live = QPushButton("En vivo")
        self.btn_live.clicked.connect(lambda: self.switch_mode("live"))

        self.btn_sim = QPushButton("Simulación")
        self.btn_sim.setObjectName("ActiveMode")
        self.btn_sim.clicked.connect(lambda: self.switch_mode("simulation"))

        self.btn_audio = QPushButton("🔇 Sonido: OFF")
        self.btn_audio.clicked.connect(self.toggle_audio)

        self.btn_export = QPushButton("Exportar captura")
        self.btn_export.clicked.connect(self.export_data)

        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)
        controls.addWidget(lbl_int)
        controls.addWidget(self.cmb_int)
        controls.addSpacing(6)
        controls.addWidget(self.btn_live)
        controls.addWidget(self.btn_sim)
        controls.addSpacing(6)
        controls.addWidget(self.btn_audio)
        controls.addStretch()
        controls.addWidget(self.btn_export)

        root.addLayout(controls, stretch=0)

    def _make_inline_pill(self, label: str, value: str) -> QFrame:
        pill = QFrame()
        pill.setStyleSheet(
            "background-color: #0d152a; border: 1px solid #1e293b; border-radius: 5px; padding: 2px 8px;"
        )
        lay = QHBoxLayout(pill)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(4)
        lbl_t = QLabel(label)
        lbl_t.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: bold; border: none;")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet("font-size: 11px; color: #ffffff; font-weight: bold; border: none;")
        lay.addWidget(lbl_t)
        lay.addWidget(lbl_v)
        pill.val_label = lbl_v  # type: ignore
        return pill

    def on_preset_changed(self, name: str):
        self.source.apply_preset(name)
        if not self.source.use_npz:
            self.sld_vel.blockSignals(True)
            self.sld_amp.blockSignals(True)
            self.sld_noise.blockSignals(True)
            self.sld_vel.setValue(int(round(self.source.vel_kms)))
            self.sld_amp.setValue(int(round(self.source.amp_db)))
            self.sld_noise.setValue(int(round(self.source.noise_db * 10)))
            self.sld_vel.blockSignals(False)
            self.sld_amp.blockSignals(False)
            self.sld_noise.blockSignals(False)
            self.lbl_sld_vel.setText(f"Velocidad: {self.source.vel_kms:.1f} km/s")
            self.lbl_sld_amp.setText(f"Pico HI: {self.source.amp_db:.1f} dB")
            self.lbl_sld_noise.setText(f"Ruido: {self.source.noise_db:.2f} dB")

    def on_slider_changed(self):
        if self.source.use_npz:
            self.source.use_npz = False
        self.source.vel_kms = float(self.sld_vel.value())
        self.source.amp_db = float(self.sld_amp.value())
        self.source.noise_db = self.sld_noise.value() / 10.0

        self.lbl_sld_vel.setText(f"Velocidad: {self.source.vel_kms:.1f} km/s")
        self.lbl_sld_amp.setText(f"Pico HI: {self.source.amp_db:.1f} dB")
        self.lbl_sld_noise.setText(f"Ruido: {self.source.noise_db:.2f} dB")

    def toggle_audio(self):
        self.audio_enabled = not self.audio_enabled
        if self.audio_enabled:
            self.btn_audio.setText("🔊 Sonido: ON")
            self.btn_audio.setObjectName("ActiveMode")
        else:
            self.btn_audio.setText("🔇 Sonido: OFF")
            self.btn_audio.setObjectName("")
        self.setStyleSheet(self.styleSheet())

    def start_acquisition(self):
        self.paused = False
        self.timer.start(100)
        self.badge_state.setText("● En vivo")
        self.badge_state.setStyleSheet(
            "border: 1px solid #10b981; color: #34d399; background-color: #064e3b; "
            "padding: 3px 10px; border-radius: 10px; font-size: 10px; font-weight: bold;"
        )

    def toggle_pause(self):
        self.paused = not self.paused
        self.badge_state.setText("● Pausado" if self.paused else "● En vivo")

    def stop_acquisition(self):
        self.timer.stop()
        self.badge_state.setText("● Detenido")
        self.badge_state.setStyleSheet(
            "border: 1px solid #334155; color: #94a3b8; background-color: #0f172a; "
            "padding: 3px 10px; border-radius: 10px; font-size: 10px; font-weight: bold;"
        )

    def switch_mode(self, mode: str):
        if mode == "live":
            QMessageBox.information(
                self,
                "Modo En Vivo (HackRF One)",
                "Para usar el modo En Vivo conecta el receptor HackRF One y enciende GNU Radio.\nSeguimos en modo Simulación."
            )
            return
        self.badge_mode.setText("SIMULACIÓN")

    def change_integration(self, idx: int):
        valores = [1.0, 5.0, 15.0]
        self.integration_s = valores[idx]
        self.source.integration_time_s = self.integration_s
        self.pill_int.val_label.setText(f"{self.integration_s:.1f} s")

    def process_next_frame(self):
        if self.paused:
            return

        raw_block = self.source.read_spectrum()
        raw_block.integration_time_s = self.integration_s
        pkt = process_hi_spectrum(raw_block)
        self.export_helper.last_packet = pkt

        freqs = np.asarray(pkt.frequencies_mhz, dtype=float)
        vels = np.asarray(pkt.velocities_kms, dtype=float)
        power = np.asarray(pkt.power_db, dtype=float)
        m = pkt.metrics

        vel_pico = m.peak_radial_velocity_kms
        freq_pico = m.peak_frequency_mhz
        snr = m.snr_db
        flujo = m.integrated_flux
        pwr_pico = float(np.max(power))

        self.lbl_big_vel.setText(f"{vel_pico:.1f}")
        self.lbl_big_freq.setText(f"{freq_pico:.3f}")
        self.lbl_big_pwr.setText(f"{pwr_pico:.1f}")
        self.lbl_sub_snr.setText(f"SNR: {snr:.2f} dB  |  Flujo: {flujo:.1f}")

        if vel_pico < -1.0:
            self.pill_dir.setText("◄ ACERCAMIENTO")
            self.pill_dir.setStyleSheet(
                "border: 1px solid #0284c7; color: #38bdf8; background-color: #0c243c; "
                "border-radius: 5px; padding: 2px 8px; font-size: 10px; font-weight: bold;"
            )
        elif vel_pico > 1.0:
            self.pill_dir.setText("ALEJAMIENTO ►")
            self.pill_dir.setStyleSheet(
                "border: 1px solid #dc2626; color: #f87171; background-color: #3b0d11; "
                "border-radius: 5px; padding: 2px 8px; font-size: 10px; font-weight: bold;"
            )
        else:
            self.pill_dir.setText("● EN REPOSO")

        self.freq_curve.setData(freqs, power)
        self.freq_dot.setData([freq_pico], [pwr_pico])
        self.freq_vline.setValue(freq_pico)
        self.freq_tag.setText(f"{freq_pico:.3f} MHz")
        self.freq_tag.setPos(freq_pico, pwr_pico)

        self.vel_curve.setData(vels, power)
        self.vel_dot.setData([vel_pico], [pwr_pico])
        self.vel_vline.setValue(vel_pico)
        self.vel_tag.setText(f"{vel_pico:.1f} km/s")
        self.vel_tag.setPos(vel_pico, pwr_pico)

        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1, :] = power
        self.wf_img.setImage(self.waterfall_data.T, autoLevels=True)
        f_min, f_max = float(freqs[0]), float(freqs[-1])
        self.wf_img.setRect(f_min, 0, f_max - f_min, self.waterfall_rows)

        if self.audio_enabled:
            chunk = generate_realtime_chunk(pkt, frame_duration_s=0.1)
            self.audio_stream.write(chunk)

    def export_data(self):
        pkt = self.export_helper.last_packet
        if pkt is None:
            QMessageBox.warning(self, "Aviso", "Aún no hay datos capturados para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Exportar captura HI", "captura_hi", "Todos los archivos (*)")
        if path:
            try:
                csv_p = f"{path}.csv"
                data_matrix = np.column_stack((pkt.frequencies_mhz, pkt.velocities_kms, pkt.power_db))
                np.savetxt(
                    csv_p,
                    data_matrix,
                    delimiter=",",
                    header="frequency_mhz,radial_velocity_kms,power_db",
                    comments="",
                )

                json_p = f"{path}.json"
                payload = {
                    "timestamp": pkt.timestamp,
                    "mode": pkt.mode,
                    "center_freq_mhz": pkt.center_freq_mhz,
                    "integration_time_s": pkt.integration_time_s,
                    "metrics": {
                        "peak_radial_velocity_kms": pkt.metrics.peak_radial_velocity_kms,
                        "peak_frequency_mhz": pkt.metrics.peak_frequency_mhz,
                        "integrated_flux": pkt.metrics.integrated_flux,
                        "snr_db": pkt.metrics.snr_db,
                    },
                    "frequencies_mhz": np.asarray(pkt.frequencies_mhz).tolist(),
                    "velocities_kms": np.asarray(pkt.velocities_kms).tolist(),
                    "power_db": np.asarray(pkt.power_db).tolist(),
                }
                with open(json_p, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

                wav_p = f"{path}.wav"
                export_simulation_wav(pkt, wav_p, duration_s=4.0)

                QMessageBox.information(
                    self,
                    "Exportado",
                    f"Archivos guardados correctamente:\n• {csv_p}\n• {json_p}\n• {wav_p}",
                )
            except Exception as e:
                QMessageBox.warning(self, "Aviso", str(e))

    def closeEvent(self, event):
        self.timer.stop()
        self.audio_stream.stop()
        self.audio_stream.close()
        event.accept()


def launch_ui():
    app = QApplication(sys.argv)
    win = RadioTelescopioWindow()
    win.showMaximized()
    win.start_acquisition()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_ui()
