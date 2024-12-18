"""
Módulo que define las clases base para la extracción de características.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List

class BaseFeatureExtractor(ABC):
    """
    Clase base abstracta para todos los extractores de características.
    
    Esta clase define la interfaz común que deben implementar todos los
    extractores de características específicos.
    """
    
    @abstractmethod
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Método abstracto para extraer características de una señal.
        
        Args:
            signal (np.ndarray): Señal de entrada (1D o 2D)
            **kwargs: Argumentos adicionales específicos del extractor
            
        Returns:
            Dict[str, float]: Diccionario con las características extraídas
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Retorna los nombres de las características que extrae este extractor.
        
        Returns:
            List[str]: Lista de nombres de características
        """
        pass
    
    def validate_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Valida y prepara la señal de entrada.
        
        Args:
            signal (np.ndarray): Señal de entrada
            
        Returns:
            np.ndarray: Señal validada y preparada
            
        Raises:
            ValueError: Si la señal no tiene el formato correcto
        """
        if signal.ndim > 2:
            raise ValueError("La señal debe ser 1D o 2D")
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            
        return signal
