import sys
import json
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
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
    QDialog,
)
from src.contracts import HI_REST_FREQ_MHZ, SPEED_OF_LIGHT_KMS, RawSpectrumBlock
from src.acquisition.hackrf_source import HackRFSource
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
        if cfg is None:
            if self.npz_source is not None:
                self.use_npz = True
            return
        self.use_npz = False
        self.vel_kms = cfg["vel_kms"]
        self.amp_db = cfg["amp_db"]
        self.sigma_mhz = cfg["sigma_mhz"]
        self.noise_db = cfg["noise_db"]

    def read_spectrum(self) -> RawSpectrumBlock:
        if self.use_npz and self.npz_source is not None:
            return self.npz_source.read_spectrum()

        sdr_dome = -85.0 - 4.5 * ((self.freqs - HI_REST_FREQ_MHZ) ** 2)
        # Ruido instantáneo por frame (el suavizado real de 1s/5s/15s lo hace el buffer deque de la ventana)
        noise = np.random.normal(0.0, self.noise_db, len(self.freqs))

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
        self.setWindowTitle("RADIOTELESCOPIO: La luz que no podemos ver — HI 21 cm")
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
        self.mode = "simulation"  # "simulation" o "live"
        self.live_source = None
        self.history_buffer: deque = deque(maxlen=300)

        self.audio_enabled = False
        try:
            self.audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            self.audio_stream.start()
        except Exception:
            self.audio_stream = None

        # Busca grabacion_noche_estrellas.npz O real_hackrf_raw_block.npz en data_samples/
        samples_dir = Path(__file__).resolve().parents[2] / "data_samples"
        rec_path = samples_dir / "grabacion_noche_estrellas.npz"
        if not rec_path.exists():
            rec_path = samples_dir / "real_hackrf_raw_block.npz"

        npz_src = RecordedNPZSource(rec_path, integration_time_s=self.integration_s) if rec_path.exists() else None
        self.source = ConfigurableSimulationSource(npz_source=npz_src, integration_time_s=self.integration_s)
        self.export_helper = SpectrumAcquisitionWorker(mode="simulation")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self._build_ui()
        self.on_preset_changed(self.cmb_preset.currentText())

    @staticmethod
    def _is_nde_logo(path: Path) -> bool:
        """Detecta si el archivo corresponde al logo de Noche de las Estrellas (ej. LOGO NDE.png)."""
        stem = path.stem.lower()
        return any(k in stem for k in ("nde", "noche", "estrella"))

    @staticmethod
    def _autocrop_and_enhance(img: QImage, is_jpg: bool, is_nde: bool) -> QImage:
        """
        1. Quita fondo blanco en JPGs.
        2. Recorta automáticamente todos los márgenes transparentes vacíos (autocrop)
           para que el logo aproveche el 100% de la altura asignada.
        3. Si el logo tiene trazos negros/muy oscuros sobre fondo transparente, los pasa a blanco.
        """
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        w, h = img.width(), img.height()

        if is_jpg:
            for y in range(h):
                for x in range(w):
                    c = img.pixelColor(x, y)
                    if c.red() > 235 and c.green() > 235 and c.blue() > 235:
                        c.setAlpha(0)
                        img.setPixelColor(x, y, c)

        # Buscar la caja delimitadora (bounding box) real de los píxeles visibles
        min_x, min_y, max_x, max_y = w, h, 0, 0
        lum_sum, count = 0.0, 0

        for y in range(h):
            for x in range(w):
                c = img.pixelColor(x, y)
                if c.alpha() > 25:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    lum_sum += 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
                    count += 1

        if max_x >= min_x and max_y >= min_y:
            img = img.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

        # Si el logo no es el de NDE y es demasiado oscuro (trazos negros), invertirlo a claro
        avg_lum = (lum_sum / count) if count > 0 else 255.0
        if not is_nde and avg_lum < 95.0:
            cw, ch = img.width(), img.height()
            for y in range(ch):
                for x in range(cw):
                    c = img.pixelColor(x, y)
                    if c.alpha() > 25:
                        c.setRed(255 - c.red())
                        c.setGreen(255 - c.green())
                        c.setBlue(255 - c.blue())
                        img.setPixelColor(x, y, c)

        return img

    def _build_logos_bar(self, logo_height: int = 52, nde_logo_height: int = 52) -> QHBoxLayout:
        """Carga, recorta márgenes vacíos y escala los logos de assets/logos/."""
        logos_layout = QHBoxLayout()
        logos_layout.setContentsMargins(6, 0, 6, 0)
        logos_layout.setSpacing(12)

        logos_dir = Path(__file__).resolve().parents[2] / "assets" / "logos"
        if not logos_dir.exists():
            return logos_layout

        archivos = [
            p for p in logos_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg")
        ]
        archivos.sort(key=lambda p: (0 if self._is_nde_logo(p) else 1, p.name.lower()))

        for idx, logo_path in enumerate(archivos):
            raw_img = QImage(str(logo_path))
            if raw_img.isNull():
                continue

            is_nde = self._is_nde_logo(logo_path) or idx == 0
            is_jpg = logo_path.suffix.lower() in (".jpg", ".jpeg")
            clean_img = self._autocrop_and_enhance(raw_img, is_jpg=is_jpg, is_nde=is_nde)

            pix = QPixmap.fromImage(clean_img)
            if not pix.isNull():
                lbl = QLabel()
                target_h = nde_logo_height if is_nde else logo_height
                scaled_pix = pix.scaledToHeight(
                    target_h, Qt.TransformationMode.SmoothTransformation
                )
                lbl.setPixmap(scaled_pix)
                lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                lbl.setStyleSheet(
                    "background-color: transparent; border: none; padding: 0px 2px;"
                )
                lbl.setToolTip(logo_path.stem.lstrip("0123456789_- ").replace("_", " ").upper())
                logos_layout.addWidget(lbl)

        return logos_layout

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)

        # ================= ENCABEZADO COMPACTO + COORDENADAS + LOGOS =================
        header = QHBoxLayout()
        header.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.setContentsMargins(0, 0, 0, 0)

        top_title_row = QHBoxLayout()
        top_title_row.setSpacing(8)
        lbl_title = QLabel("RADIOTELESCOPIO:")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: 800; letter-spacing: 0.8px; color: #ffffff;")
        lbl_sub = QLabel("HI – 21 cm")
        lbl_sub.setStyleSheet("font-size: 11px; color: #f59e0b; font-weight: bold;")
        top_title_row.addWidget(lbl_title)
        top_title_row.addWidget(lbl_sub)
        top_title_row.addStretch()

        lbl_tagline = QLabel("La luz que no podemos ver")
        lbl_tagline.setStyleSheet("font-size: 11px; font-style: italic; color: #38bdf8; font-weight: 600;")

        title_box.addLayout(top_title_row)
        title_box.addWidget(lbl_tagline)

        header.addLayout(title_box)
        header.addSpacing(8)

        header.addWidget(self._make_inline_pill("AR:", "18h 36m 00s"))
        header.addWidget(self._make_inline_pill("DEC:", "−08° 30′ 00″"))
        self.pill_int = self._make_inline_pill("Integración:", "5.0 s")
        header.addWidget(self.pill_int)

        header.addStretch()
        header.addLayout(self._build_logos_bar(logo_height=28, nde_logo_height=42))
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

        root.addLayout(row_plots, stretch=5)

        # ================= FILA DE MÉTRICAS COMPACTA =================
        row_metrics = QHBoxLayout()
        row_metrics.setSpacing(8)

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

        # ================= CASCADA · HISTORIAL DEL ESPECTRO =================
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
        self.cmb_preset.setCurrentIndex(1)  # Iniciar en Nube acercándose por defecto
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

        self.btn_credits = QPushButton("👥 Créditos")
        self.btn_credits.clicked.connect(self.show_credits_dialog)

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
        controls.addWidget(self.btn_credits)
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

    def show_credits_dialog(self):
        """Muestra una ventana emergente con los créditos de los 3 equipos participantes."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Créditos — RADIOTELESCOPIO: La luz que no podemos ver")
        dlg.setMinimumWidth(700)
        dlg.setStyleSheet("""
            QDialog { background-color: #060b19; color: #f8fafc; }
            QFrame#TeamCard { background-color: #0d152a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
            QLabel { border: none; background: transparent; }
        """)

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)

        hdr = QLabel("RADIOTELESCOPIO: La luz que no podemos ver")
        hdr.setStyleSheet("font-size: 17px; font-weight: 800; color: #ffffff;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel(
            "Observación de la línea de hidrógeno neutro a 21 cm (1420.405 MHz)\n"
            "Instituto de Astronomía UNAM · Noche de las Estrellas · Astrolab"
        )
        sub.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: 600;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(hdr)
        lay.addWidget(sub)

        teams_row = QHBoxLayout()
        teams_row.setSpacing(12)

        #TODO Agregar los nombres de los integrantes de los equipos 2 y 3
        equipos = [
            (
                "1. Software, DSP e Interfaz",
                "#f59e0b",
                [
                    "Ing. Leonardo Garces",
                    "Alan Vladimir Alonso Sánchez",
                    "Ricardo Ramirez Flores",
                    "Katia Denisse Rayon Martínez",
                ],
            ),
            (
                "2. Instrumentación y RF / Antena",
                "#38bdf8",
                [
                    "Ejemplo:",
                    "Integrante 1 (Equipo 2)",
                    "Integrante 2 (Equipo 2)",
                    "Integrante 3 (Equipo 2)",
                ],
            ),
            (
                "3. Mecánica, Montura y Divulgación",
                "#34d399",
                [
                    "Integrante 1 (Equipo 3)",
                    "Integrante 2 (Equipo 3)",
                    "Integrante 3 (Equipo 3)",
                ],
            ),
        ]

        for titulo_eq, color_hex, miembros in equipos:
            card = QFrame()
            card.setObjectName("TeamCard")
            c_lay = QVBoxLayout(card)
            c_lay.setSpacing(6)

            lbl_eq = QLabel(titulo_eq)
            lbl_eq.setWordWrap(True)
            lbl_eq.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color_hex};")
            c_lay.addWidget(lbl_eq)

            for m in miembros:
                lbl_m = QLabel(f"• {m}")
                lbl_m.setWordWrap(True)
                lbl_m.setStyleSheet("font-size: 11px; color: #e2e8f0;")
                c_lay.addWidget(lbl_m)

            c_lay.addStretch()
            teams_row.addWidget(card)

        lay.addLayout(teams_row)

        btn_close = QPushButton("Cerrar")
        btn_close.setFixedWidth(120)
        btn_close.clicked.connect(dlg.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        dlg.exec()

    def on_preset_changed(self, name: str):
        self.history_buffer.clear()
        self.source.apply_preset(name)

        if self.source.use_npz:
            self.badge_mode.setText("GRABACIÓN (.NPZ)")
            self.sld_vel.setEnabled(False)
            self.sld_amp.setEnabled(False)
            self.sld_noise.setEnabled(False)
            return

        if self.mode == "simulation":
            self.badge_mode.setText("SIMULACIÓN")
        self.sld_vel.setEnabled(True)
        self.sld_amp.setEnabled(True)
        self.sld_noise.setEnabled(True)

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
        self.history_buffer.clear()
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
        self.history_buffer.clear()
        if mode == "live":
            try:
                if self.live_source is None:
                    self.live_source = HackRFSource(
                        num_bins=1024,
                        sample_rate_mhz=2.0,
                        integration_time_s=self.integration_s,
                    )
                _ = self.live_source.read_spectrum()
                self.mode = "live"
                self.badge_mode.setText("EN VIVO (HACKRF)")
                self.btn_live.setObjectName("ActiveMode")
                self.btn_sim.setObjectName("")
                self.setStyleSheet(self.styleSheet())
            except Exception as exc:
                if self.live_source is not None:
                    self.live_source.close()
                    self.live_source = None
                QMessageBox.warning(
                    self,
                    "HackRF / GNU Radio no detectado",
                    f"No se recibió señal en tcp://127.0.0.1:5555.\n"
                    f"Asegúrate de correr gnuradio/receptor_hi_flowgraph.py primero.\n\nDetalle: {exc}",
                )
            return

        if self.live_source is not None:
            self.live_source.close()
            self.live_source = None
        self.mode = "simulation"
        self.badge_mode.setText("GRABACIÓN (.NPZ)" if self.source.use_npz else "SIMULACIÓN")
        self.btn_sim.setObjectName("ActiveMode")
        self.btn_live.setObjectName("")
        self.setStyleSheet(self.styleSheet())

    def change_integration(self, idx: int):
        valores = [1.0, 5.0, 15.0]
        self.integration_s = valores[idx]
        self.source.integration_time_s = self.integration_s
        self.pill_int.val_label.setText(f"{self.integration_s:.1f} s")

    def process_next_frame(self):
        if self.paused:
            return

        try:
            if self.mode == "live" and self.live_source is not None:
                raw_block = self.live_source.read_spectrum()
            else:
                raw_block = self.source.read_spectrum()
        except Exception:
            self.switch_mode("simulation")
            return

        raw_block.integration_time_s = self.integration_s

        target_frames = max(1, int(self.integration_s * 10))
        self.history_buffer.append(np.asarray(raw_block.power_db, dtype=float).copy())
        while len(self.history_buffer) > target_frames:
            self.history_buffer.popleft()

        lin_stack = [10.0 ** (p / 10.0) for p in self.history_buffer]
        raw_block.power_db = 10.0 * np.log10(np.mean(lin_stack, axis=0) + 1e-12)

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
        pwr_pico = m.peak_power_db

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
            self.pill_dir.setStyleSheet(
                "border: 1px solid #10b981; color: #34d399; background-color: #064e3b; "
                "border-radius: 5px; padding: 2px 8px; font-size: 10px; font-weight: bold;"
            )

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

        if self.audio_enabled and self.audio_stream is not None:
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
                csv_p, json_p = self.export_helper.export_last_frame(path)
                wav_p = f"{Path(path).with_suffix('')}.wav"
                export_simulation_wav(pkt, wav_p, duration_s=4.0)

                QMessageBox.information(
                    self,
                    "Exportado",
                    f"Archivos guardados correctamente:\n- {csv_p}\n- {json_p}\n- {wav_p}",
                )
            except Exception as e:
                QMessageBox.warning(self, "Aviso", str(e))

    def closeEvent(self, event):
        self.timer.stop()
        if self.live_source is not None:
            self.live_source.close()
        if self.audio_stream is not None:
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