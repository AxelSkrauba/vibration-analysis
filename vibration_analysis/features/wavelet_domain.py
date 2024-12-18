"""
Módulo para extracción de características en el dominio wavelet.
Incluye implementaciones para CWT (Continuous Wavelet Transform) y
DWT (Discrete Wavelet Transform).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pywt
from scipy import stats
from .base import BaseFeatureExtractor

class WaveletFeatureExtractor(BaseFeatureExtractor):
    """
    Clase base para extracción de características basadas en wavelets.
    
    Esta clase proporciona la funcionalidad básica para el análisis
    wavelet y la extracción de características, sirviendo como base
    para implementaciones específicas (CWT y DWT).
    """
    
    # Características disponibles para análisis wavelet
    AVAILABLE_FEATURES = {
        'wavelet_energy': 'Energía total de los coeficientes wavelet',
        'wavelet_entropy': 'Entropía de Shannon de los coeficientes',
        'wavelet_variance': 'Varianza de los coeficientes',
        'wavelet_kurtosis': 'Kurtosis de los coeficientes',
        'wavelet_skewness': 'Asimetría de los coeficientes',
        'wavelet_max': 'Valor máximo de los coeficientes',
        'wavelet_mean': 'Valor medio de los coeficientes',
        'wavelet_std': 'Desviación estándar de los coeficientes',
        'relative_energy': 'Energía relativa por nivel/escala',
        'energy_ratio': 'Ratio de energía entre niveles/escalas'
    }
    
    def __init__(self, wavelet: str = 'morl', features: Optional[List[str]] = None):
        """
        Inicializa el extractor de características wavelet.
        
        Args:
            wavelet: Nombre de la wavelet a utilizar (ver pywt.wavelist())
            features: Lista de características a extraer. Si es None, se extraen todas.
        """
        self.wavelet = wavelet
        self.features = features if features is not None else list(self.AVAILABLE_FEATURES.keys())
        self._validate_wavelet()
    
    def _validate_wavelet(self):
        """Valida que la wavelet especificada sea válida."""
        available_wavelets = pywt.wavelist()
        if self.wavelet not in available_wavelets:
            raise ValueError(
                f"Wavelet '{self.wavelet}' no válida. "
                f"Opciones disponibles: {available_wavelets}"
            )
    
    def _compute_entropy(self, coeffs: np.ndarray) -> float:
        """Calcula la entropía de Shannon de los coeficientes."""
        coeffs = np.abs(coeffs)
        norm_coeffs = coeffs / np.sum(coeffs)
        entropy = -np.sum(norm_coeffs * np.log2(norm_coeffs + np.finfo(float).eps))
        return entropy
    
    def _compute_energy(self, coeffs: np.ndarray) -> float:
        """Calcula la energía total de los coeficientes."""
        return np.sum(np.abs(coeffs)**2)
    
    def _compute_relative_energy(self, coeffs_list: List[np.ndarray]) -> np.ndarray:
        """Calcula la energía relativa para cada nivel/escala."""
        energies = np.array([self._compute_energy(c) for c in coeffs_list])
        total_energy = np.sum(energies)
        return energies / total_energy if total_energy > 0 else energies
    
    def _compute_basic_stats(self, coeffs: np.ndarray) -> Dict[str, float]:
        """Calcula estadísticas básicas de los coeficientes."""
        return {
            'wavelet_mean': np.mean(coeffs),
            'wavelet_std': np.std(coeffs),
            'wavelet_max': np.max(np.abs(coeffs)),
            'wavelet_kurtosis': stats.kurtosis(coeffs.flatten()),
            'wavelet_skewness': stats.skew(coeffs.flatten()),
            'wavelet_variance': np.var(coeffs)
        }
    
    def get_feature_names(self) -> List[str]:
        """Retorna la lista de nombres de características disponibles."""
        return self.features

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Método abstracto que debe ser implementado por las clases derivadas.
        """
        raise NotImplementedError("Las clases derivadas deben implementar este método")

class CWTFeatureExtractor(WaveletFeatureExtractor):
    """
    Extractor de características basado en la Transformada Wavelet Continua (CWT).
    """
    
    def __init__(self,
                 wavelet: str = 'morl',
                 scales: Optional[np.ndarray] = None,
                 features: Optional[List[str]] = None):
        """
        Inicializa el extractor CWT.
        
        Args:
            wavelet: Tipo de wavelet a utilizar
            scales: Escalas para CWT. Si es None, se generan automáticamente
            features: Lista de características a extraer
        """
        super().__init__(wavelet=wavelet, features=features)
        self.scales = scales
    
    def _generate_scales(self, signal_length: int) -> np.ndarray:
        """Genera escalas apropiadas si no se especifican."""
        if self.scales is None:
            num_scales = min(32, signal_length // 2)
            self.scales = np.logspace(0, np.log10(signal_length/2), num_scales)
        return self.scales
    
    def _compute_cwt(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula la CWT de la señal."""
        scales = self._generate_scales(len(signal))
        coeffs, freqs = pywt.cwt(signal, scales, self.wavelet)
        return coeffs, freqs
    
    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extrae características basadas en CWT de la señal.
        
        Args:
            signal: Señal de entrada (1D o 2D)
            
        Returns:
            Diccionario con las características extraídas
        """
        # Manejar señales multicanal
        if signal.ndim > 1:
            features_per_channel = [
                self.extract(channel) for channel in signal.T
            ]
            # Combinar características de todos los canales
            combined_features = {}
            for i, channel_features in enumerate(features_per_channel):
                for name, value in channel_features.items():
                    combined_features[f"{name}_ch{i+1}"] = value
            return combined_features
        
        # Calcular CWT
        coeffs, _ = self._compute_cwt(signal)
        
        # Inicializar diccionario de características
        features = {}
        
        # Extraer características solicitadas
        if 'wavelet_energy' in self.features:
            features['wavelet_energy'] = self._compute_energy(coeffs)
            
        if 'wavelet_entropy' in self.features:
            features['wavelet_entropy'] = self._compute_entropy(coeffs)
            
        if 'relative_energy' in self.features:
            rel_energies = self._compute_relative_energy([coeffs[i, :] for i in range(len(self.scales))])
            for i, energy in enumerate(rel_energies):
                features[f'relative_energy_scale_{i+1}'] = energy
        
        # Extraer estadísticas básicas si están solicitadas
        basic_stats = ['wavelet_mean', 'wavelet_std', 'wavelet_max',
                      'wavelet_kurtosis', 'wavelet_skewness', 'wavelet_variance']
        
        for stat in basic_stats:
            if stat in self.features:
                features.update({
                    stat: self._compute_basic_stats(coeffs)[stat]
                })
        
        return features

class DWTFeatureExtractor(WaveletFeatureExtractor):
    """
    Extractor de características basado en la Transformada Wavelet Discreta (DWT).
    """
    
    def __init__(self,
                 wavelet: str = 'db4',
                 level: Optional[int] = None,
                 features: Optional[List[str]] = None):
        """
        Inicializa el extractor DWT.
        
        Args:
            wavelet: Tipo de wavelet a utilizar
            level: Nivel de descomposición. Si es None, se calcula automáticamente
            features: Lista de características a extraer
        """
        super().__init__(wavelet=wavelet, features=features)
        self.level = level
    
    def _get_max_level(self, signal_length: int) -> int:
        """Calcula el nivel máximo de descomposición posible."""
        return pywt.dwt_max_level(signal_length, pywt.Wavelet(self.wavelet).dec_len)
    
    def _compute_dwt(self, signal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Calcula la DWT de la señal."""
        if self.level is None:
            self.level = self._get_max_level(len(signal))
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        return coeffs[0], coeffs[1:]  # (aproximación, [detalles])
    
    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extrae características basadas en DWT de la señal.
        
        Args:
            signal: Señal de entrada (1D o 2D)
            
        Returns:
            Diccionario con las características extraídas
        """
        # Manejar señales multicanal
        if signal.ndim > 1:
            features_per_channel = [
                self.extract(channel) for channel in signal.T
            ]
            # Combinar características de todos los canales
            combined_features = {}
            for i, channel_features in enumerate(features_per_channel):
                for name, value in channel_features.items():
                    combined_features[f"{name}_ch{i+1}"] = value
            return combined_features
        
        # Calcular DWT
        approx, details = self._compute_dwt(signal)
        
        # Inicializar diccionario de características
        features = {}
        
        # Características de aproximación
        if 'wavelet_energy' in self.features:
            features['approx_energy'] = self._compute_energy(approx)
            for i, detail in enumerate(details):
                features[f'detail_energy_level_{i+1}'] = self._compute_energy(detail)
        
        if 'wavelet_entropy' in self.features:
            features['approx_entropy'] = self._compute_entropy(approx)
            for i, detail in enumerate(details):
                features[f'detail_entropy_level_{i+1}'] = self._compute_entropy(detail)
        
        if 'relative_energy' in self.features:
            all_coeffs = [approx] + details
            rel_energies = self._compute_relative_energy(all_coeffs)
            features['approx_relative_energy'] = rel_energies[0]
            for i, energy in enumerate(rel_energies[1:]):
                features[f'detail_relative_energy_level_{i+1}'] = energy
        
        # Estadísticas básicas para aproximación y detalles
        basic_stats = ['wavelet_mean', 'wavelet_std', 'wavelet_max',
                      'wavelet_kurtosis', 'wavelet_skewness', 'wavelet_variance']
        
        for stat in basic_stats:
            if stat in self.features:
                # Estadísticas de aproximación
                approx_stats = self._compute_basic_stats(approx)
                features[f'approx_{stat}'] = approx_stats[stat]
                
                # Estadísticas de detalles por nivel
                for i, detail in enumerate(details):
                    detail_stats = self._compute_basic_stats(detail)
                    features[f'detail_{stat}_level_{i+1}'] = detail_stats[stat]
        
        return features
