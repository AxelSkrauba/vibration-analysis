"""
Módulo de extracción de características

Este módulo contiene las clases y funciones necesarias para la extracción
de características de señales de vibración en diferentes dominios.
"""

from .base import BaseFeatureExtractor
from .time_domain import TimeFeatureExtractor
from .frequency_domain import FrequencyFeatureExtractor
from .wavelet_domain import CWTFeatureExtractor, DWTFeatureExtractor

__all__ = [
    'BaseFeatureExtractor',
    'TimeFeatureExtractor',
    'FrequencyFeatureExtractor',
    'CWTFeatureExtractor',
    'DWTFeatureExtractor'
]
