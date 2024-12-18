"""
Módulo para procesamiento básico de señales.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional

class SignalProcessor:
    """
    Clase para procesamiento básico de señales de vibración.
    
    Implementa métodos comunes de preprocesamiento como filtrado,
    normalización y segmentación de señales.
    """
    
    @staticmethod
    def normalize(signal: np.ndarray, method: str = 'robust') -> np.ndarray:
        """
        Normaliza la señal usando diferentes métodos.
        
        Args:
            signal (np.ndarray): Señal de entrada
            method (str): Método de normalización ('zscore', 'minmax', 'robust')
            
        Returns:
            np.ndarray: Señal normalizada
        """
        if method == 'zscore':
            return (signal - np.mean(signal)) / np.std(signal)
        elif method == 'minmax':
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        elif method == 'robust':
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            return (signal - median) / mad
        else:
            raise ValueError(f"Método de normalización '{method}' no reconocido")
    
    @staticmethod
    def filter_signal(signal: np.ndarray, 
                     fs: float,
                     lowcut: Optional[float] = None,
                     highcut: Optional[float] = None,
                     order: int = 4) -> np.ndarray:
        """
        Aplica un filtro Butterworth a la señal.
        
        Args:
            signal (np.ndarray): Señal de entrada
            fs (float): Frecuencia de muestreo
            lowcut (float, opcional): Frecuencia de corte inferior
            highcut (float, opcional): Frecuencia de corte superior
            order (int): Orden del filtro
            
        Returns:
            np.ndarray: Señal filtrada
        """
        nyq = 0.5 * fs
        
        if lowcut and highcut:
            # Filtro pasa banda
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
        elif lowcut:
            # Filtro pasa altos
            low = lowcut / nyq
            b, a = butter(order, low, btype='high')
        elif highcut:
            # Filtro pasa bajos
            high = highcut / nyq
            b, a = butter(order, high, btype='low')
        else:
            return signal
            
        return filtfilt(b, a, signal)
    
    @staticmethod
    def segment_signal(signal: np.ndarray,
                      window_size: int,
                      overlap: float = 0.5) -> np.ndarray:
        """
        Segmenta la señal en ventanas con solapamiento.
        
        Args:
            signal (np.ndarray): Señal de entrada
            window_size (int): Tamaño de la ventana en muestras
            overlap (float): Porcentaje de solapamiento entre ventanas (0-1)
            
        Returns:
            np.ndarray: Array de ventanas
        """
        if not 0 <= overlap < 1:
            raise ValueError("El solapamiento debe estar entre 0 y 1")
            
        step = int(window_size * (1 - overlap))
        windows = []
        
        for i in range(0, len(signal) - window_size + 1, step):
            windows.append(signal[i:i + window_size])
            
        return np.array(windows)
    
    @staticmethod
    def remove_dc(signal: np.ndarray) -> np.ndarray:
        """
        Elimina el componente DC de la señal.
        
        Args:
            signal (np.ndarray): Señal de entrada
            
        Returns:
            np.ndarray: Señal sin componente DC
        """
        return signal - np.mean(signal)
