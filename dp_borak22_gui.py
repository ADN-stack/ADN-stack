"""dp_borak22_gui.py
======================

Application pédagogique permettant de simuler des décharges partielles (DP) dans
une huile isolante Borak 22 et de visualiser les signaux associés (courant,
lumière, chute de tension et champ rayonné). L'interface graphique s'appuie sur
PyQt5 et Matplotlib, tandis que NumPy et SciPy sont utilisés pour la génération
et l'analyse fréquentielle des signaux.

Utilisation
----------

$ python dp_borak22_gui.py

Dépendances principales :

* Python 3.10+
* PyQt5
* Matplotlib
* NumPy
* SciPy (facultatif mais recommandé pour les FFT/DCT rapides)

Si SciPy n'est pas disponible, l'application reste fonctionnelle et propose des
implémentations alternatives (plus lentes) ainsi qu'un message invitant à
l'installation de ``scipy`` via ``pip install scipy``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    # SciPy fournit des implémentations efficaces de la FFT (rfft) et de la DCT.
    from scipy.fft import dct as scipy_dct, idct as scipy_idct, rfft, rfftfreq

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - dépend de l'environnement utilisateur
    SCIPY_AVAILABLE = False
    scipy_dct = None
    scipy_idct = None
    rfft = None
    rfftfreq = None

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt
except Exception as exc:  # pragma: no cover - dépend de l'environnement utilisateur
    raise SystemExit(
        "PyQt5 est requis pour exécuter cette application. Installez-le avec "
        "pip install pyqt5"
    ) from exc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

# ---------------------------------------------------------------------------
# Constantes et paramètres par défaut
# ---------------------------------------------------------------------------

CHANNELS = ["courant", "lumiere", "delta_v", "champ"]
CHANNEL_LABELS = {
    "courant": "Courant i(t) [A]",
    "lumiere": "Lumière L(t) [cd]",
    "delta_v": "Chute de tension ΔV(t) [V]",
    "champ": "Champ rayonné E(t) [V/m]",
}
CHANNEL_COLORS = {
    "courant": "tabblue",
    "lumiere": "taborange",
    "delta_v": "tabgreen",
    "champ": "tabred",
}

MAX_ANALYSIS_SAMPLES = 20000 if SCIPY_AVAILABLE else 2000


@dataclass
class SimulationParameters:
    """Regroupe les paramètres configurables de la simulation.

    Cette classe est sérialisable en JSON pour l'enregistrement de presets.
    Les unités sont précisées dans les attributs.
    """

    duration_ms: float = 50.0
    fs_MHz: float = 20.0
    lambda_rate: float = 500.0  # impulsions par seconde
    width_min_us: float = 0.5
    width_max_us: float = 5.0
    amplitude_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "courant": (2.0, 8.0),
            "lumiere": (1.0, 5.0),
            "delta_v": (50.0, 200.0),
            "champ": (5.0, 20.0),
        }
    )
    noise_percent: Dict[str, float] = field(
        default_factory=lambda: {
            "courant": 3.0,
            "lumiere": 5.0,
            "delta_v": 2.0,
            "champ": 4.0,
        }
    )
    ring_frequency_MHz: float = 10.0
    ring_damping: float = 0.05
    seed: Optional[int] = 42

    def to_json(self) -> str:
        """Sérialise les paramètres en JSON."""

        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(data: str) -> "SimulationParameters":
        """Reconstruit un ensemble de paramètres à partir d'une chaîne JSON."""

        payload = json.loads(data)
        return SimulationParameters(**payload)


# ---------------------------------------------------------------------------
# Fonctions utilitaires liées à la simulation physique
# ---------------------------------------------------------------------------


def random_lognormal(min_val: float, max_val: float) -> float:
    """Génère une valeur suivant une loi log-normale bornée.

    La loi log-normale permet de représenter les amplitudes stochastiques des DP
    avec une forte dissymétrie (beaucoup de petites impulsions et quelques
    grandes). La valeur est recadrée dans l'intervalle [min_val, max_val].
    """

    mu = math.log(math.sqrt(min_val * max_val))
    sigma = 0.4
    value = np.random.lognormal(mean=mu, sigma=sigma)
    return float(np.clip(value, min_val, max_val))


def simulate_impulses(params: SimulationParameters) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, float]]:
    """Simule les décharges partielles et retourne les signaux.

    Parameters
    ----------
    params : SimulationParameters
        Paramètres configurables de la simulation.

    Returns
    -------
    time : np.ndarray
        Vecteur temporel (secondes).
    signals : dict
        Dictionnaire ``canal -> signal`` (numpy arrays).
    dp_times : np.ndarray
        Instants (secondes) des décharges simulées.
    info : dict
        Informations agrégées (énergie, taux, etc.).

    Notes
    -----
    * Les occurrences des DP suivent un processus de Poisson de taux ``lambda``.
    * Les formes d'impulsion sont inspirées des réponses caractéristiques des
      capteurs correspondants.
    * Chaque canal reçoit un bruit blanc gaussien d'écart-type ajustable.
    """

    duration_s = params.duration_ms / 1e3
    fs = params.fs_MHz * 1e6
    n_samples = int(max(1, duration_s * fs))
    time_vect = np.arange(n_samples) / fs

    # Préparation des tableaux de signaux
    signals = {ch: np.zeros(n_samples, dtype=float) for ch in CHANNELS}

    # Génération des occurrences via un processus de Poisson
    rate = max(1e-3, params.lambda_rate)
    expected_count = max(1, int(duration_s * rate * 1.5))
    inter_arrivals = np.random.exponential(1.0 / rate, size=expected_count * 2)
    dp_times = []
    t_cursor = 0.0
    for delta in inter_arrivals:
        t_cursor += delta
        if t_cursor >= duration_s:
            break
        dp_times.append(t_cursor)
    dp_times = np.array(dp_times, dtype=float)

    # Paramètres de largeur
    width_min = params.width_min_us / 1e6
    width_max = params.width_max_us / 1e6
    width_min = max(width_min, 0.1e-6)
    width_max = max(width_max, width_min * 1.2)

    ring_freq = params.ring_frequency_MHz * 1e6
    damping = max(1e-4, params.ring_damping)

    # Création de chaque impulsion
    energies = []
    for t0 in dp_times:
        idx0 = int(t0 * fs)
        if idx0 >= n_samples:
            continue
        pulse_width = random_lognormal(width_min, width_max)
        window_length = int(min(n_samples - idx0, max(10, pulse_width * fs * 6)))
        if window_length <= 1:
            continue
        t_rel = np.arange(window_length) / fs

        amp_current = random_lognormal(*params.amplitude_ranges["courant"])
        tau_rise = pulse_width * 0.25
        tau_decay = pulse_width
        impulse = amp_current * (
            np.exp(-t_rel / tau_decay) - np.exp(-t_rel / tau_rise)
        )
        impulse /= np.max(np.abs(impulse) + 1e-12)
        impulse *= amp_current
        signals["courant"][idx0 : idx0 + window_length] += impulse

        # Énergie approximative de l'impulsion (J ~ ∫ i^2 dt)
        energies.append(float(np.trapz(impulse**2, dx=1 / fs)))

        # Lumière : proportionnelle à |i(t)| filtré par un RC (rémanence)
        amp_light = random_lognormal(*params.amplitude_ranges["lumiere"])
        tau_light = pulse_width * 3
        light_pulse = amp_light * (np.exp(-t_rel / tau_light))
        signals["lumiere"][idx0 : idx0 + window_length] += light_pulse

        # ΔV : impulsion négative suivie d'une récupération linéaire
        amp_v = random_lognormal(*params.amplitude_ranges["delta_v"])
        drop = -amp_v * np.exp(-t_rel / (pulse_width * 0.6))
        recovery = amp_v * (t_rel / (pulse_width * 4.0))
        voltage_pulse = drop + recovery
        signals["delta_v"][idx0 : idx0 + window_length] += voltage_pulse

        # Champ rayonné : sinusoïde amortie centrée sur f0
        amp_e = random_lognormal(*params.amplitude_ranges["champ"])
        envelope = np.exp(-damping * ring_freq * t_rel)
        rf = np.sin(2 * np.pi * ring_freq * t_rel)
        field_pulse = amp_e * envelope * rf
        signals["champ"][idx0 : idx0 + window_length] += field_pulse

    # Ajout de bruit blanc gaussien
    for ch in CHANNELS:
        max_amp = params.amplitude_ranges[ch][1]
        sigma = params.noise_percent[ch] / 100.0 * max_amp
        signals[ch] += np.random.normal(0.0, sigma, size=n_samples)

    info = {
        "count": int(len(dp_times)),
        "energy": float(np.mean(energies) if energies else 0.0),
        "fs": fs,
    }
    return time_vect, signals, dp_times, info


# ---------------------------------------------------------------------------
# Analyse fréquentielle : FFT et DCT
# ---------------------------------------------------------------------------


def compute_fft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule la FFT réelle d'un signal.

    Parameters
    ----------
    x : np.ndarray
        Signal temporel.
    fs : float
        Fréquence d'échantillonnage en Hz.

    Returns
    -------
    freq : np.ndarray
        Axe fréquentiel en MHz pour faciliter la lecture.
    magnitude : np.ndarray
        Module du spectre (valeurs positives).
    """

    if SCIPY_AVAILABLE and rfft is not None and rfftfreq is not None:
        spectrum = rfft(x)
        freq = rfftfreq(x.size, d=1.0 / fs)
    else:  # fallback numpy
        spectrum = np.fft.rfft(x)
        freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    magnitude = np.abs(spectrum)
    return freq / 1e6, magnitude


def compute_dct(x: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Calcule la DCT-II d'un signal avec éventuellement une compression."""

    if SCIPY_AVAILABLE and scipy_dct is not None:
        coeffs = scipy_dct(x, type=2, norm="ortho")
    else:
        # Implémentation générique de la DCT-II basée sur la définition.
        n = x.size
        k_idx = np.arange(n)
        factor = np.pi / n
        coeffs = []
        for m in range(n):
            coeff = np.sum(x * np.cos(factor * (k_idx + 0.5) * m)) * math.sqrt(2 / n)
            if m == 0:
                coeff *= 1 / math.sqrt(2)
            coeffs.append(coeff)
        coeffs = np.array(coeffs)
    if k is not None:
        coeffs_k = np.copy(coeffs)
        coeffs_k[k:] = 0.0
        return coeffs_k
    return coeffs


def idct_reconstruct(coeffs: np.ndarray) -> np.ndarray:
    """Reconstruit un signal à partir de coefficients de DCT-II."""

    if SCIPY_AVAILABLE and scipy_idct is not None:
        return scipy_idct(coeffs, type=2, norm="ortho")
    # Implémentation inverse basique
    n = coeffs.size
    x = np.zeros(n)
    factor = np.pi / n
    for k_idx in range(n):
        summation = 0.0
        for m in range(n):
            alpha = math.sqrt(1 / n) if m == 0 else math.sqrt(2 / n)
            summation += alpha * coeffs[m] * math.cos(factor * (k_idx + 0.5) * m)
        x[k_idx] = summation
    return x


def estimate_snr(x: np.ndarray) -> float:
    """Estime le rapport signal/bruit (SNR) à partir d'un signal.

    La méthode est basée sur une séparation grossière entre énergie basse
    fréquence et haute fréquence. On considère que la moyenne glissante représente
    le signal utile tandis que la différence constitue le bruit.
    """

    if x.size < 4:
        return 0.0
    window = max(5, x.size // 200)
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="same")
    noise = x - trend
    power_signal = np.mean(trend**2)
    power_noise = np.mean(noise**2) + 1e-12
    snr = 10 * np.log10(power_signal / power_noise)
    return float(np.clip(snr, -60.0, 80.0))


# ---------------------------------------------------------------------------
# Widgets utilitaires (boîte rétractable, slider/spin synchronisés)
# ---------------------------------------------------------------------------


class CollapsibleBox(QtWidgets.QWidget):
    """Boîte repliable utilisant un bouton d'entête."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.toggled.connect(self.on_toggled)

        self.content_area = QtWidgets.QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        lay.setContentsMargins(0, 0, 0, 0)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight", duration=150)
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight", duration=150)
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight", duration=150)
        )

    def setContentLayout(self, layout: QtWidgets.QLayout) -> None:
        """Définit le contenu interne de la boîte repliable."""

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.content_area.setWidget(widget)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + layout.sizeHint().height())

    def on_toggled(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward if checked else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()


class SliderSpinWidget(QtWidgets.QWidget):
    """Association d'un slider et d'un spin box synchronisés."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        orientation: Qt.Orientation = Qt.Horizontal,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.decimals = decimals
        self.minimum = minimum
        self.maximum = maximum
        self.step = step

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QtWidgets.QSlider(orientation)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(round((maximum - minimum) / step)))
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self._slider_changed)

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.valueChanged.connect(self._spin_changed)

        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

    def _slider_changed(self, value: int) -> None:
        real_value = self.minimum + value * self.step
        real_value = max(self.minimum, min(self.maximum, real_value))
        self.spin.blockSignals(True)
        self.spin.setValue(real_value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(real_value)

    def _spin_changed(self, value: float) -> None:
        slider_value = int(round((value - self.minimum) / self.step))
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)


# ---------------------------------------------------------------------------
# Widget de contrôle (dock) permettant d'ajuster la simulation
# ---------------------------------------------------------------------------


class ControlsPane(QtWidgets.QWidget):
    """Dock widget regroupant tous les contrôles utilisateur."""

    parametersChanged = QtCore.pyqtSignal(SimulationParameters)
    generateRequested = QtCore.pyqtSignal()
    startStopRequested = QtCore.pyqtSignal(bool)
    resetRequested = QtCore.pyqtSignal()
    loadPresetRequested = QtCore.pyqtSignal()
    savePresetRequested = QtCore.pyqtSignal()
    toggleMarkersRequested = QtCore.pyqtSignal(bool)

    def __init__(self, params: SimulationParameters, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.params = params
        self.setToolTip("Paramètres de génération des décharges partielles Borak 22")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_simulation_section())
        layout.addWidget(self._build_rate_section())
        layout.addWidget(self._build_amplitude_section())
        layout.addWidget(self._build_width_section())
        layout.addWidget(self._build_noise_section())
        layout.addWidget(self._build_rf_section())

        layout.addStretch(1)

    # -- sections -----------------------------------------------------------------

    def _build_simulation_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Simulation")
        form = QtWidgets.QFormLayout()

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 200.0)
        self.duration_spin.setSuffix(" ms")
        self.duration_spin.setValue(self.params.duration_ms)
        self.duration_spin.valueChanged.connect(self._emit_parameters)
        self.duration_spin.setToolTip("Durée de la trace temporelle en millisecondes")
        form.addRow("Durée", self.duration_spin)

        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(1.0, 100.0)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setSuffix(" MHz")
        self.fs_spin.setValue(self.params.fs_MHz)
        self.fs_spin.valueChanged.connect(self._emit_parameters)
        self.fs_spin.setToolTip("Fréquence d'échantillonnage (MHz)")
        form.addRow("Fs", self.fs_spin)

        seed_layout = QtWidgets.QHBoxLayout()
        self.seed_edit = QtWidgets.QLineEdit(str(self.params.seed if self.params.seed is not None else ""))
        self.seed_edit.setPlaceholderText("seed")
        self.seed_edit.editingFinished.connect(self._emit_parameters)
        seed_layout.addWidget(self.seed_edit)
        random_button = QtWidgets.QPushButton("Aléatoire")
        random_button.setToolTip("Choisir une graine aléatoire pour la simulation")
        random_button.clicked.connect(self._random_seed)
        seed_layout.addWidget(random_button)
        form.addRow("Seed", seed_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.generate_button = QtWidgets.QPushButton("Générer")
        self.generate_button.clicked.connect(self.generateRequested.emit)
        button_layout.addWidget(self.generate_button)

        self.play_button = QtWidgets.QPushButton("Lecture")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._play_toggled)
        self.play_button.setToolTip("Lancer ou arrêter la génération continue")
        button_layout.addWidget(self.play_button)

        reset_button = QtWidgets.QPushButton("Réinitialiser")
        reset_button.clicked.connect(self.resetRequested.emit)
        button_layout.addWidget(reset_button)
        form.addRow("Actions", button_layout)

        box.setContentLayout(form)
        return box

    def _build_rate_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Taux de DP")
        form = QtWidgets.QFormLayout()

        self.lambda_slider = SliderSpinWidget(10.0, 3000.0, 10.0, 0)
        self.lambda_slider.setValue(self.params.lambda_rate)
        self.lambda_slider.valueChanged.connect(self._emit_parameters)
        self.lambda_slider.setToolTip("Taux de Poisson λ (impulsions par seconde)")
        form.addRow("λ [imp/s]", self.lambda_slider)

        box.setContentLayout(form)
        return box

    def _build_amplitude_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Amplitude (min / max)")
        layout = QtWidgets.QVBoxLayout()
        self.amplitude_sliders: Dict[str, Tuple[SliderSpinWidget, SliderSpinWidget]] = {}
        for ch in CHANNELS:
            min_widget = SliderSpinWidget(0.1, 2000.0, 0.1, 1)
            max_widget = SliderSpinWidget(0.1, 4000.0, 0.1, 1)
            min_widget.setValue(self.params.amplitude_ranges[ch][0])
            max_widget.setValue(self.params.amplitude_ranges[ch][1])
            min_widget.valueChanged.connect(lambda v, c=ch: self._update_amplitude(c, 0, v))
            max_widget.valueChanged.connect(lambda v, c=ch: self._update_amplitude(c, 1, v))
            group = QtWidgets.QGroupBox(CHANNEL_LABELS[ch])
            form = QtWidgets.QFormLayout()
            form.addRow("Min", min_widget)
            form.addRow("Max", max_widget)
            group.setLayout(form)
            layout.addWidget(group)
            self.amplitude_sliders[ch] = (min_widget, max_widget)
        box.setContentLayout(layout)
        return box

    def _build_width_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Largeur d'impulsion")
        form = QtWidgets.QFormLayout()
        self.width_min = SliderSpinWidget(0.1, 50.0, 0.1, 1)
        self.width_min.setValue(self.params.width_min_us)
        self.width_min.valueChanged.connect(self._emit_parameters)
        self.width_min.setToolTip("Largeur minimale des impulsions (µs)")
        form.addRow("Min [µs]", self.width_min)
        self.width_max = SliderSpinWidget(0.2, 100.0, 0.1, 1)
        self.width_max.setValue(self.params.width_max_us)
        self.width_max.valueChanged.connect(self._emit_parameters)
        self.width_max.setToolTip("Largeur maximale des impulsions (µs)")
        form.addRow("Max [µs]", self.width_max)
        box.setContentLayout(form)
        return box

    def _build_noise_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Bruit blanc (σ en %)")
        layout = QtWidgets.QVBoxLayout()
        self.noise_sliders: Dict[str, SliderSpinWidget] = {}
        for ch in CHANNELS:
            widget = SliderSpinWidget(0.0, 50.0, 0.5, 1)
            widget.setValue(self.params.noise_percent[ch])
            widget.valueChanged.connect(lambda v, c=ch: self._update_noise(c, v))
            widget.setToolTip("Écart-type du bruit blanc en pourcentage de l'amplitude max")
            form = QtWidgets.QFormLayout()
            form.addRow(CHANNEL_LABELS[ch], widget)
            container = QtWidgets.QWidget()
            container.setLayout(form)
            layout.addWidget(container)
            self.noise_sliders[ch] = widget
        box.setContentLayout(layout)
        return box

    def _build_rf_section(self) -> CollapsibleBox:
        box = CollapsibleBox("Sonnerie RF du champ")
        form = QtWidgets.QFormLayout()
        self.ring_freq = SliderSpinWidget(1.0, 80.0, 0.5, 1)
        self.ring_freq.setValue(self.params.ring_frequency_MHz)
        self.ring_freq.valueChanged.connect(self._emit_parameters)
        self.ring_freq.setToolTip("Fréquence centrale de la sonnerie RF (MHz)")
        form.addRow("f₀ [MHz]", self.ring_freq)

        self.ring_damp = SliderSpinWidget(0.001, 0.5, 0.001, 3)
        self.ring_damp.setValue(self.params.ring_damping)
        self.ring_damp.valueChanged.connect(self._emit_parameters)
        self.ring_damp.setToolTip("Facteur d'amortissement de la sinusoïde amortie")
        form.addRow("ζ", self.ring_damp)

        markers_button = QtWidgets.QPushButton("Marqueurs de DP")
        markers_button.setCheckable(True)
        markers_button.toggled.connect(self.toggleMarkersRequested.emit)
        form.addRow("Options", markers_button)

        box.setContentLayout(form)
        return box

    # -- interaction ---------------------------------------------------------------

    def _play_toggled(self, checked: bool) -> None:
        self.play_button.setText("Stop" if checked else "Lecture")
        self.startStopRequested.emit(checked)

    def _random_seed(self) -> None:
        value = random.randint(0, 999999)
        self.seed_edit.setText(str(value))
        self._emit_parameters()

    def _update_amplitude(self, channel: str, index: int, value: float) -> None:
        min_val, max_val = self.params.amplitude_ranges[channel]
        if index == 0:
            min_val = min(value, max_val - 0.1)
        else:
            max_val = max(value, min_val + 0.1)
        self.params.amplitude_ranges[channel] = (min_val, max_val)
        self._emit_parameters()

    def _update_noise(self, channel: str, value: float) -> None:
        self.params.noise_percent[channel] = value
        self._emit_parameters()

    def _emit_parameters(self) -> None:
        self.params.duration_ms = self.duration_spin.value()
        self.params.fs_MHz = self.fs_spin.value()
        self.params.lambda_rate = self.lambda_slider.value()
        self.params.width_min_us = self.width_min.value()
        self.params.width_max_us = self.width_max.value()
        self.params.ring_frequency_MHz = self.ring_freq.value()
        self.params.ring_damping = self.ring_damp.value()
        try:
            seed_text = self.seed_edit.text().strip()
            self.params.seed = int(seed_text) if seed_text else None
        except ValueError:
            self.params.seed = None
        self.parametersChanged.emit(self.params)


# ---------------------------------------------------------------------------
# PlotPane : gère l'affichage matplotlib (temps / FFT / DCT)
# ---------------------------------------------------------------------------


class PlotPane(QtWidgets.QWidget):
    """Zone centrale contenant les différents graphiques."""

    windowSelected = QtCore.pyqtSignal(float, float)
    peakClicked = QtCore.pyqtSignal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self._create_time_tab()
        self._create_fft_tab()
        self._create_dct_tab()

        self.dp_times = np.array([])
        self._marker_enabled = False
        self._span_selector: Optional[SpanSelector] = None
        self._last_time_vect = np.array([])
        self._last_signals: Dict[str, np.ndarray] = {}
        self._last_fft_window: Optional[Tuple[int, int]] = None
        self._last_dct_window: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    def _create_time_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.time_fig = Figure(figsize=(6, 6))
        self.time_canvas = FigureCanvas(self.time_fig)
        self.time_axes = [self.time_fig.add_subplot(4, 1, i + 1) for i in range(4)]
        self.time_fig.tight_layout(pad=1.5)
        layout.addWidget(self.time_canvas)
        self.tabs.addTab(widget, "Temps")
        self.time_canvas.mpl_connect("button_press_event", self._on_time_click)
        self._span_selector = SpanSelector(
            self.time_axes[0],
            onselect=self._on_span_selected,
            direction="horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="tab:blue"),
        )

    def _create_fft_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Échelle"))
        self.fft_scale_combo = QtWidgets.QComboBox()
        self.fft_scale_combo.addItems(["Linéaire", "Log10"])
        controls_layout.addWidget(self.fft_scale_combo)
        self.fft_scale_combo.currentIndexChanged.connect(self._on_fft_scale_changed)
        controls_layout.addStretch(1)
        layout.addLayout(controls_layout)
        self.fft_fig = Figure(figsize=(6, 6))
        self.fft_canvas = FigureCanvas(self.fft_fig)
        self.fft_axes = [self.fft_fig.add_subplot(4, 1, i + 1) for i in range(4)]
        self.fft_fig.tight_layout(pad=1.5)
        layout.addWidget(self.fft_canvas)
        self.tabs.addTab(widget, "FFT")

    def _create_dct_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("Coefficients conservés"))
        self.dct_keep_spin = QtWidgets.QSpinBox()
        self.dct_keep_spin.setRange(1, 5000)
        self.dct_keep_spin.setValue(200)
        control_layout.addWidget(self.dct_keep_spin)
        self.dct_keep_spin.setToolTip(
            "Nombre de coefficients DCT conservés pour la reconstruction inverse"
        )
        self.dct_keep_spin.valueChanged.connect(self._on_dct_keep_changed)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

        self.dct_fig = Figure(figsize=(6, 6))
        self.dct_canvas = FigureCanvas(self.dct_fig)
        self.dct_axes = [self.dct_fig.add_subplot(4, 1, i + 1) for i in range(4)]
        self.dct_twin_axes = [ax.twinx() for ax in self.dct_axes]
        self.dct_fig.tight_layout(pad=1.5)
        layout.addWidget(self.dct_canvas)
        self.tabs.addTab(widget, "DCT")

    # ------------------------------------------------------------------
    def update_signals(
        self,
        time_vect: np.ndarray,
        signals: Dict[str, np.ndarray],
        dp_times: np.ndarray,
        params: SimulationParameters,
    ) -> None:
        """Met à jour les trois onglets avec les nouveaux signaux."""

        self.dp_times = dp_times
        self._last_time_vect = time_vect
        self._last_signals = signals
        self._update_time_domain(time_vect, signals)
        self.update_frequency_domain(time_vect, signals)
        self.update_dct(time_vect, signals)

    def _update_time_domain(self, time_vect: np.ndarray, signals: Dict[str, np.ndarray]) -> None:
        for ax, ch in zip(self.time_axes, CHANNELS):
            ax.clear()
            ax.plot(time_vect * 1e3, signals[ch], color=CHANNEL_COLORS[ch], lw=0.9, label=CHANNEL_LABELS[ch])
            if self._marker_enabled and self.dp_times.size:
                ymin, ymax = ax.get_ylim()
                ax.vlines(
                    self.dp_times * 1e3,
                    ymin=ymin,
                    ymax=ymax,
                    colors="gray",
                    linestyles="dotted",
                )
                ax.set_ylim(ymin, ymax)
            ax.set_ylabel(CHANNEL_LABELS[ch])
            ax.grid(True, which="both", ls=":")
        self.time_axes[-1].set_xlabel("Temps [ms]")
        self.time_fig.tight_layout(pad=1.2)
        self.time_canvas.draw_idle()

    def update_frequency_domain(self, time_vect: np.ndarray, signals: Dict[str, np.ndarray], window: Optional[Tuple[int, int]] = None) -> None:
        if window is None:
            slice_ = slice(None)
        else:
            slice_ = slice(window[0], window[1])
        if time_vect.size < 2:
            return
        fs = 1.0 / (time_vect[1] - time_vect[0])
        segment_len = signals[CHANNELS[0]][slice_].size
        if segment_len == 0:
            return
        step = max(1, int(math.ceil(segment_len / MAX_ANALYSIS_SAMPLES)))
        fs_eff = fs / step
        for ax, ch in zip(self.fft_axes, CHANNELS):
            ax.clear()
            segment = signals[ch][slice_]
            if step > 1:
                segment = segment[::step]
            freq, mag = compute_fft(segment, fs_eff)
            if self.fft_scale_combo.currentIndex() == 1:
                mag = 20 * np.log10(mag + 1e-12)
                ax.set_ylabel("|FFT| [dB]")
            else:
                ax.set_ylabel("|FFT|")
            ax.plot(freq, mag, color=CHANNEL_COLORS[ch])
            ax.set_xlim(0, fs_eff / 1e6 / 2)
            ax.set_title(f"Spectre {CHANNEL_LABELS[ch]}")
            ax.grid(True, which="both", ls=":")
        self.fft_axes[-1].set_xlabel("Fréquence [MHz]")
        self.fft_fig.tight_layout(pad=1.2)
        self.fft_canvas.draw_idle()
        self._last_fft_window = window

    def update_dct(
        self,
        time_vect: np.ndarray,
        signals: Dict[str, np.ndarray],
        window: Optional[Tuple[int, int]] = None,
    ) -> None:
        if window is None:
            slice_ = slice(None)
        else:
            slice_ = slice(window[0], window[1])
        keep = self.dct_keep_spin.value()
        base_segment = signals[CHANNELS[0]][slice_]
        if base_segment.size == 0:
            return
        step = max(1, int(math.ceil(base_segment.size / MAX_ANALYSIS_SAMPLES)))
        for ax, twin, ch in zip(self.dct_axes, self.dct_twin_axes, CHANNELS):
            ax.clear()
            twin.clear()
            segment = signals[ch][slice_]
            if step > 1:
                segment = segment[::step]
            if segment.size == 0:
                continue
            coeffs = compute_dct(segment)
            keep_local = min(keep, coeffs.size)
            truncated = np.copy(coeffs)
            if truncated.size > keep_local:
                truncated[keep_local:] = 0.0
            coeff_idx = np.arange(coeffs.size)
            reconstruction = idct_reconstruct(truncated)
            rec_x = np.linspace(0, reconstruction.size - 1, reconstruction.size)
            ax.plot(coeff_idx, coeffs, color=CHANNEL_COLORS[ch], alpha=0.4, label="Coefficients")
            ax.plot(coeff_idx, truncated, color="black", lw=0.8, label="Conservés")
            twin.plot(rec_x, reconstruction, color="tabpurple", lw=0.8)
            twin.set_ylabel("Amplitude reconstruite")
            twin.grid(False)
            ax.set_title(f"DCT {CHANNEL_LABELS[ch]}")
            ax.grid(True, ls=":")
            ax.legend(loc="upper right")
        self.dct_axes[-1].set_xlabel("Indice des coefficients")
        self.dct_fig.tight_layout(pad=1.2)
        self.dct_canvas.draw_idle()
        self._last_dct_window = window

    def _on_fft_scale_changed(self) -> None:
        if not hasattr(self, "_last_time_vect"):
            return
        window = getattr(self, "_last_fft_window", None)
        self.update_frequency_domain(self._last_time_vect, self._last_signals, window)

    def _on_dct_keep_changed(self) -> None:
        if not hasattr(self, "_last_time_vect"):
            return
        window = getattr(self, "_last_dct_window", None)
        self.update_dct(self._last_time_vect, self._last_signals, window)

    # ------------------------------------------------------------------
    def toggle_markers(self, enabled: bool) -> None:
        self._marker_enabled = enabled

    # ------------------------------------------------------------------
    def _on_span_selected(self, x0: float, x1: float) -> None:
        if x0 == x1:
            return
        start_ms, end_ms = sorted([x0, x1])
        self.windowSelected.emit(start_ms / 1e3, end_ms / 1e3)

    def _on_time_click(self, event) -> None:
        if event.xdata is None or event.inaxes not in self.time_axes:
            return
        t_click = event.xdata / 1e3
        self.peakClicked.emit(t_click)


# ---------------------------------------------------------------------------
# Simulator : encapsule les calculs et gère la graine aléatoire
# ---------------------------------------------------------------------------


class Simulator(QtCore.QObject):
    """Classe utilitaire orchestrant la génération des signaux."""

    simulationReady = QtCore.pyqtSignal(np.ndarray, dict, np.ndarray, dict)

    def __init__(self, params: SimulationParameters, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.params = params
        self.last_result = None

    def set_parameters(self, params: SimulationParameters) -> None:
        self.params = params

    def run_simulation(self) -> None:
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
            random.seed(self.params.seed)
        time_vect, signals, dp_times, info = simulate_impulses(self.params)
        self.last_result = (time_vect, signals, dp_times, info)
        self.simulationReady.emit(time_vect, signals, dp_times, info)


# ---------------------------------------------------------------------------
# Fenêtre principale regroupant tout l'écosystème
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    """Fenêtre principale de l'application Borak 22 DP."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simulateur DP Borak 22")
        self.resize(1400, 900)

        self.params = SimulationParameters()
        self.simulator = Simulator(self.params)

        self.plot_pane = PlotPane()
        self.setCentralWidget(self.plot_pane)

        self.controls = ControlsPane(self.params)
        self.controls.parametersChanged.connect(self._update_parameters)
        self.controls.generateRequested.connect(self.simulator.run_simulation)
        self.controls.startStopRequested.connect(self._toggle_timer)
        self.controls.resetRequested.connect(self._reset_defaults)
        self.controls.toggleMarkersRequested.connect(self._toggle_markers)

        self.dock = QtWidgets.QDockWidget("Paramètres")
        self.dock.setWidget(self.controls)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self._create_actions()
        self._create_menus()
        self._create_toolbar()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.simulator.run_simulation)
        self.timer.setInterval(50)

        self.simulator.simulationReady.connect(self._on_simulation_ready)
        self.plot_pane.windowSelected.connect(self._on_window_selected)
        self.plot_pane.peakClicked.connect(self._on_peak_clicked)

        if not SCIPY_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "SciPy non disponible",
                "SciPy n'a pas été détecté. Les calculs FFT/DCT utiliseront des\n"
                "implémentations alternatives plus lentes.\n"
                "Installez scipy via : pip install scipy",
            )

        self.simulator.run_simulation()

        self._apply_dark_palette()

    # ------------------------------------------------------------------
    def _create_actions(self) -> None:
        self.export_csv_act = QtWidgets.QAction("Exporter CSV", self)
        self.export_csv_act.triggered.connect(self._export_csv)

        self.export_png_act = QtWidgets.QAction("Exporter PNG", self)
        self.export_png_act.triggered.connect(self._export_png)

        self.copy_clipboard_act = QtWidgets.QAction("Copier l'image", self)
        self.copy_clipboard_act.triggered.connect(self._copy_to_clipboard)

        self.load_preset_act = QtWidgets.QAction("Charger preset", self)
        self.load_preset_act.triggered.connect(self._load_preset)

        self.save_preset_act = QtWidgets.QAction("Enregistrer preset", self)
        self.save_preset_act.triggered.connect(self._save_preset)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("Fichier")
        file_menu.addAction(self.export_csv_act)
        file_menu.addAction(self.export_png_act)
        file_menu.addAction(self.copy_clipboard_act)
        file_menu.addSeparator()
        file_menu.addAction(self.load_preset_act)
        file_menu.addAction(self.save_preset_act)
        file_menu.addSeparator()
        quit_action = QtWidgets.QAction("Quitter", self)
        quit_action.triggered.connect(QtWidgets.qApp.quit)
        file_menu.addAction(quit_action)

    def _create_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Actions")
        toolbar.addAction(self.export_png_act)
        toolbar.addAction(self.export_csv_act)
        toolbar.addAction(self.copy_clipboard_act)
        toolbar.addAction(self.load_preset_act)
        toolbar.addAction(self.save_preset_act)
        self.addToolBar(toolbar)

    # ------------------------------------------------------------------
    def _update_parameters(self, params: SimulationParameters) -> None:
        self.params = params
        self.simulator.set_parameters(params)

    def _toggle_timer(self, start: bool) -> None:
        if start:
            self.timer.start()
        else:
            self.timer.stop()

    def _reset_defaults(self) -> None:
        self.params = SimulationParameters()
        self.timer.stop()
        self.controls.play_button.setChecked(False)
        self.controls.params = self.params
        self.controls.duration_spin.setValue(self.params.duration_ms)
        self.controls.fs_spin.setValue(self.params.fs_MHz)
        self.controls.lambda_slider.setValue(self.params.lambda_rate)
        self.controls.width_min.setValue(self.params.width_min_us)
        self.controls.width_max.setValue(self.params.width_max_us)
        self.controls.ring_freq.setValue(self.params.ring_frequency_MHz)
        self.controls.ring_damp.setValue(self.params.ring_damping)
        self.controls.seed_edit.setText(str(self.params.seed))
        for ch in CHANNELS:
            min_widget, max_widget = self.controls.amplitude_sliders[ch]
            min_widget.setValue(self.params.amplitude_ranges[ch][0])
            max_widget.setValue(self.params.amplitude_ranges[ch][1])
            self.controls.noise_sliders[ch].setValue(self.params.noise_percent[ch])
        self.simulator.set_parameters(self.params)
        self.simulator.run_simulation()

    def _toggle_markers(self, enabled: bool) -> None:
        self.plot_pane.toggle_markers(enabled)
        if self.simulator.last_result:
            self._on_simulation_ready(*self.simulator.last_result)

    def _on_simulation_ready(
        self,
        time_vect: np.ndarray,
        signals: Dict[str, np.ndarray],
        dp_times: np.ndarray,
        info: Dict[str, float],
    ) -> None:
        self.plot_pane.update_signals(time_vect, signals, dp_times, self.params)
        snr = estimate_snr(signals["courant"])
        status_text = (
            f"DP: {info['count']} | Energie moyenne ≈ {info['energy']:.2e} J | "
            f"SNR courant ≈ {snr:.1f} dB"
        )
        self.status.showMessage(status_text)

    def _on_window_selected(self, start_s: float, end_s: float) -> None:
        if not self.simulator.last_result:
            return
        time_vect, signals, _, _ = self.simulator.last_result
        idx_start = max(0, int(start_s * self.params.fs_MHz * 1e6))
        idx_end = min(len(time_vect), int(end_s * self.params.fs_MHz * 1e6))
        if idx_end - idx_start < 10:
            return
        window = (idx_start, idx_end)
        self.plot_pane.update_frequency_domain(time_vect, signals, window)
        self.plot_pane.update_dct(time_vect, signals, window)

    def _on_peak_clicked(self, t_click: float) -> None:
        if not self.simulator.last_result:
            return
        _, signals, dp_times, _ = self.simulator.last_result
        if dp_times.size == 0:
            return
        idx = np.argmin(np.abs(dp_times - t_click))
        dp_time = dp_times[idx]
        fs = self.params.fs_MHz * 1e6
        sample = int(dp_time * fs)
        # Estimation grossière de l'amplitude et de la largeur locale
        window = slice(max(0, sample - 50), min(sample + 50, len(signals["courant"])))
        segment = signals["courant"][window]
        amplitude = float(np.max(segment) - np.min(segment))
        # Largeur estimée via demi-maximum
        half = np.max(segment) * 0.5
        indices = np.where(segment >= half)[0]
        if indices.size:
            width_samples = indices[-1] - indices[0]
            width_us = width_samples / fs * 1e6
        else:
            width_us = 0.0
        QtWidgets.QMessageBox.information(
            self,
            "Impulsion",
            (
                f"Instant : {dp_time*1e3:.3f} ms\n"
                f"Amplitude approx. : {amplitude:.2f} A\n"
                f"Largeur approx. : {width_us:.2f} µs"
            ),
        )

    def _export_csv(self) -> None:
        if not self.simulator.last_result:
            return
        time_vect, signals, _, _ = self.simulator.last_result
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter CSV", "dp_borak22.csv", "CSV (*.csv)")
        if not path:
            return
        data = np.column_stack([
            time_vect,
            signals["courant"],
            signals["lumiere"],
            signals["delta_v"],
            signals["champ"],
        ])
        header = "time_s,courant_A,lumiere_cd,deltaV_V,champ_Vm"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        self.status.showMessage(f"CSV exporté : {path}", 5000)

    def _export_png(self) -> None:
        current_widget = self.plot_pane.tabs.currentWidget()
        if current_widget is None:
            return
        default_name = "figure.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter PNG", default_name, "PNG (*.png)")
        if not path:
            return
        if self.plot_pane.tabs.currentIndex() == 0:
            self.plot_pane.time_fig.savefig(path, dpi=200)
        elif self.plot_pane.tabs.currentIndex() == 1:
            self.plot_pane.fft_fig.savefig(path, dpi=200)
        else:
            self.plot_pane.dct_fig.savefig(path, dpi=200)
        self.status.showMessage(f"PNG exporté : {path}", 5000)

    def _copy_to_clipboard(self) -> None:
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        if self.plot_pane.tabs.currentIndex() == 0:
            self.plot_pane.time_fig.savefig(buffer, format="png")
        elif self.plot_pane.tabs.currentIndex() == 1:
            self.plot_pane.fft_fig.savefig(buffer, format="png")
        else:
            self.plot_pane.dct_fig.savefig(buffer, format="png")
        image = QtGui.QImage.fromData(buffer.data())
        QtWidgets.QApplication.clipboard().setImage(image)
        self.status.showMessage("Figure copiée dans le presse-papiers", 4000)

    def _load_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Charger preset", "", "Preset JSON (*.json)"
        )
        if not path:
            return
        try:
            data = Path(path).read_text(encoding="utf8")
            params = SimulationParameters.from_json(data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible de charger le preset : {exc}",
            )
            return
        self.params = params
        self.simulator.set_parameters(params)
        self.controls.params = params
        self.controls.duration_spin.setValue(params.duration_ms)
        self.controls.fs_spin.setValue(params.fs_MHz)
        self.controls.lambda_slider.setValue(params.lambda_rate)
        self.controls.width_min.setValue(params.width_min_us)
        self.controls.width_max.setValue(params.width_max_us)
        self.controls.ring_freq.setValue(params.ring_frequency_MHz)
        self.controls.ring_damp.setValue(params.ring_damping)
        self.controls.seed_edit.setText(str(params.seed if params.seed is not None else ""))
        for ch in CHANNELS:
            min_widget, max_widget = self.controls.amplitude_sliders[ch]
            min_widget.setValue(params.amplitude_ranges[ch][0])
            max_widget.setValue(params.amplitude_ranges[ch][1])
            self.controls.noise_sliders[ch].setValue(params.noise_percent[ch])
        self.simulator.run_simulation()

    def _save_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer preset", "preset.json", "Preset JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.params.to_json(), encoding="utf8")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible d'enregistrer le preset : {exc}",
            )
            return
        self.status.showMessage(f"Preset enregistré : {path}", 5000)

    def _apply_dark_palette(self) -> None:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.WindowText, Qt.white)
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ToolTipBase, Qt.white)
        palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
        palette.setColor(QtGui.QPalette.Text, Qt.white)
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
        palette.setColor(QtGui.QPalette.BrightText, Qt.red)
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(70, 120, 200))
        palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
        QtWidgets.QApplication.instance().setPalette(palette)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Point d'entrée du script."""

    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("Borak")
    app.setApplicationName("Simulateur DP Borak 22")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
