"""
Módulo para extracción de características en el dominio de la frecuencia.
"""

import numpy as np
from scipy import fftpack
from typing import Dict, List, Optional, Tuple
from .base import BaseFeatureExtractor

class FrequencyFeatureExtractor(BaseFeatureExtractor):
    """
    Extractor de características en el dominio de la frecuencia.
    
    Esta clase implementa la extracción de características espectrales
    utilizando FFT y el método de Welch para estimación espectral.
    """
    
    def __init__(self, fs: float = 1000.0, low_freq_cutoff: float = 2.0, features: Optional[List[str]] = None):
        """
        Inicializa el extractor de características frecuenciales.
        
        Args:
            fs (float): Frecuencia de muestreo en Hz
            low_freq_cutoff (float): Frecuencia de corte inferior en Hz. 
                                   Frecuencias por debajo de este valor serán ignoradas
            features (List[str], opcional): Lista de características a extraer.
                Si es None, se extraen todas las características disponibles.
        """
        self.fs = fs
        self.low_freq_cutoff = low_freq_cutoff
        self._fft = None
        self._freqs = None
        self._freq_bands = None
        
        # Características disponibles
        self._available_features = {
            'spectral_centroid': self._spectral_centroid,
            'spectral_spread': self._spectral_spread,
            'spectral_rms': self._spectral_rms,
            'spectral_energy': self._spectral_energy,
            'peak_frequency': self._peak_frequency,
            'mean_frequency': self._mean_frequency
        }
        
        self._features = features if features else list(self._available_features.keys())
        
    def set_frequency_bands(self, bands: List[Tuple[float, float]]):
        """
        Establece las bandas de frecuencia para el análisis.
        
        Args:
            bands (List[Tuple[float, float]]): Lista de tuplas (freq_min, freq_max)
                definiendo los rangos de frecuencia en Hz
        """
        self._freq_bands = bands
        
    def _get_band_indices(self, band: Tuple[float, float]) -> Tuple[int, int]:
        """
        Obtiene los índices del espectro correspondientes a una banda.
        
        Args:
            band (Tuple[float, float]): (freq_min, freq_max) en Hz
            
        Returns:
            Tuple[int, int]: Índices (idx_min, idx_max)
        """
        freq_min, freq_max = band
        idx_min = np.argmin(np.abs(self._freqs - freq_min))
        idx_max = np.argmin(np.abs(self._freqs - freq_max))
        return idx_min, idx_max
        
    def _compute_band_features(self, fft_data: np.ndarray, band: Tuple[float, float]) -> Dict[str, float]:
        """
        Calcula características para una banda específica.
        
        Args:
            fft_data (np.ndarray): Datos FFT
            band (Tuple[float, float]): (freq_min, freq_max) en Hz
            
        Returns:
            Dict[str, float]: Características de la banda
        """
        idx_min, idx_max = self._get_band_indices(band)
        band_fft = fft_data[idx_min:idx_max+1]
        band_freqs = self._freqs[idx_min:idx_max+1]
        
        features = {}
        for feature_name in self._features:
            if feature_name in self._available_features:
                # Temporalmente reemplazamos los datos FFT completos con los de la banda
                original_fft = self._fft
                original_freqs = self._freqs
                self._fft = band_fft
                self._freqs = band_freqs
                
                feature_func = self._available_features[feature_name]
                features[f'{feature_name}_{int(band[0])}_{int(band[1])}Hz'] = feature_func()
                
                # Restauramos los datos FFT originales
                self._fft = original_fft
                self._freqs = original_freqs
                
        return features

    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Extrae características frecuenciales de la señal.
        
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
            # Calcular FFT y características por canal
            for i in range(n_channels):
                channel_signal = signal[:, i]
                axis_name = axes_names[i] if i < len(axes_names) else f'channel_{i}'
                
                # Calcular FFT para este canal
                self._fft = self._compute_fft(channel_signal)
                self._freqs = fftpack.fftfreq(len(channel_signal), d=1/self.fs)[:len(channel_signal)//2]
                
                # Si hay bandas definidas, calcular características por banda
                if self._freq_bands:
                    for band in self._freq_bands:
                        band_features = self._compute_band_features(self._fft, band)
                        for name, value in band_features.items():
                            features[f'{name}_{axis_name}'] = value
                
                # Extraer características para todo el espectro
                for feature_name in self._features:
                    if feature_name in self._available_features:
                        feature_func = self._available_features[feature_name]
                        features[f'{feature_name}_{axis_name}'] = feature_func()
            
            # Calcular características combinadas
            if self._freq_bands:
                for band in self._freq_bands:
                    for feature_name in self._features:
                        if feature_name in self._available_features:
                            combined_value = np.mean([
                                features[f'{feature_name}_{int(band[0])}_{int(band[1])}Hz_{axes_names[i]}'] 
                                for i in range(n_channels)
                            ])
                            features[f'{feature_name}_{int(band[0])}_{int(band[1])}Hz_combined'] = combined_value
            
            # Características combinadas para todo el espectro
            for feature_name in self._features:
                if feature_name in self._available_features:
                    combined_value = np.mean([features[f'{feature_name}_{axes_names[i]}'] 
                                           for i in range(n_channels)])
                    features[f'{feature_name}_combined'] = combined_value
        else:
            # Para señales de un solo canal
            self._fft = self._compute_fft(signal.flatten())
            self._freqs = fftpack.fftfreq(len(signal), d=1/self.fs)[:len(signal)//2]
            
            # Si hay bandas definidas, calcular características por banda
            if self._freq_bands:
                for band in self._freq_bands:
                    band_features = self._compute_band_features(self._fft, band)
                    features.update(band_features)
            
            # Características para todo el espectro
            for feature_name in self._features:
                if feature_name in self._available_features:
                    feature_func = self._available_features[feature_name]
                    features[feature_name] = feature_func()
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna los nombres de las características que extrae este extractor.
        
        Returns:
            List[str]: Lista de nombres de características
        """
        return self._features
    
    def _compute_fft(self, signal: np.ndarray) -> np.ndarray:
        """
        Calcula la FFT de la señal y aplica el filtro de bajas frecuencias.
        
        Args:
            signal (np.ndarray): Señal de entrada
            
        Returns:
            np.ndarray: Magnitud del espectro de frecuencias filtrado
        """
        # Calcular FFT
        fft = fftpack.fft(signal)
        magnitude = np.abs(fft)[:len(signal)//2]
        
        # Crear máscara para filtrar bajas frecuencias
        if self.low_freq_cutoff > 0:
            freqs = fftpack.fftfreq(len(signal), d=1/self.fs)[:len(signal)//2]
            magnitude[freqs < self.low_freq_cutoff] = 0
        
        return magnitude
    
    # Funciones de características individuales
    def _spectral_centroid(self) -> float:
        """Centroide espectral"""
        return float(np.sum(self._freqs * self._fft) / np.sum(self._fft))
    
    def _spectral_spread(self) -> float:
        """Dispersión espectral"""
        centroid = self._spectral_centroid()
        return float(np.sqrt(np.sum((self._freqs - centroid)**2 * self._fft) / np.sum(self._fft)))
    
    def _spectral_rms(self) -> float:
        """RMS espectral"""
        return float(np.sqrt(np.mean(self._fft**2)))
    
    def _spectral_energy(self) -> float:
        """Energía espectral"""
        return float(np.sum(self._fft**2))
    
    def _peak_frequency(self) -> float:
        """Frecuencia de pico"""
        return float(self._freqs[np.argmax(self._fft)])
    
    def _mean_frequency(self) -> float:
        """Frecuencia media"""
        return float(np.average(self._freqs, weights=self._fft))
