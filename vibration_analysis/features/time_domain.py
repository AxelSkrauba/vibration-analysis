"""
Módulo para extracción de características en el dominio del tiempo.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from .base import BaseFeatureExtractor

class TimeFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor de características en el dominio del tiempo.
    
    Esta clase implementa la extracción de características estadísticas
    básicas y avanzadas en el dominio del tiempo.
    """
    
    def __init__(self, features: Optional[List[str]] = None):
        """
        Inicializa el extractor de características temporales.
        
        Args:
            features (List[str], opcional): Lista de características a extraer.
                Si es None, se extraen todas las características disponibles.
        """
        self._available_features = {
            'std': self._std,
            'var': self._var,
            'rms': self._rms,
            'peak': self._peak,
            'peak_to_peak': self._peak_to_peak,
            'kurtosis': self._kurtosis,
            'skewness': self._skewness,
            'crest_factor': self._crest_factor
        }
        
        self._features = features or list(self._available_features.keys())
        
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Extrae características temporales de la señal.
        
        Args:
            signal (np.ndarray): Señal de entrada (puede ser multicanal)
            **kwargs: Argumentos adicionales
            
        Returns:
            Dict[str, float]: Diccionario con las características extraídas
        """
        signal = self.validate_signal(signal)
        features = {}
        
        # Determinar si la señal es multicanal
        n_channels = signal.shape[1] if len(signal.shape) > 1 else 1
        axes_names = ['x', 'y', 'z']
        
        if n_channels > 1:
            # Calcular características por canal
            for i in range(n_channels):
                channel_signal = signal[:, i]
                axis_name = axes_names[i] if i < len(axes_names) else f'channel_{i}'
                
                for feature_name in self._features:
                    if feature_name in self._available_features:
                        feature_func = self._available_features[feature_name]
                        # Guardar característica con nombre específico del canal
                        features[f'{feature_name}_{axis_name}'] = feature_func(channel_signal)
                        
            # Calcular características combinadas (promedio de los canales)
            for feature_name in self._features:
                if feature_name in self._available_features:
                    feature_func = self._available_features[feature_name]
                    combined_value = np.mean([features[f'{feature_name}_{axes_names[i]}'] 
                                           for i in range(n_channels)])
                    features[f'{feature_name}_combined'] = combined_value
        else:
            # Para señales de un solo canal, calcular características directamente
            for feature_name in self._features:
                if feature_name in self._available_features:
                    feature_func = self._available_features[feature_name]
                    features[feature_name] = feature_func(signal.flatten())
        
        return features
        
    def get_feature_names(self) -> List[str]:
        """
        Retorna los nombres de las características que extrae este extractor.
        
        Returns:
            List[str]: Lista de nombres de características
        """
        return self._features
    
    # Funciones de características individuales
    def _std(self, signal: np.ndarray) -> float:
        """Desviación estándar"""
        return float(np.std(signal))
    
    def _var(self, signal: np.ndarray) -> float:
        """Varianza"""
        return float(np.var(signal))
    
    def _rms(self, signal: np.ndarray) -> float:
        """Root Mean Square (RMS)"""
        return float(np.sqrt(np.mean(signal**2)))
    
    def _peak(self, signal: np.ndarray) -> float:
        """Valor pico (máximo absoluto)"""
        return float(np.max(np.abs(signal)))
    
    def _peak_to_peak(self, signal: np.ndarray) -> float:
        """Valor pico a pico"""
        return float(np.max(signal) - np.min(signal))
    
    def _kurtosis(self, signal: np.ndarray) -> float:
        """Kurtosis"""
        return float(stats.kurtosis(signal))
    
    def _skewness(self, signal: np.ndarray) -> float:
        """Asimetría (skewness)"""
        return float(stats.skew(signal))
    
    def _crest_factor(self, signal: np.ndarray) -> float:
        """Factor de cresta"""
        rms = self._rms(signal)
        if rms > 0:
            return float(self._peak(signal) / rms)
        return 0.0
