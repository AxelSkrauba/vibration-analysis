"""
Módulo para carga y preparación de datos de vibración.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from ..preprocessing.signal_processing import SignalProcessor

class DataLoader:
    """
    Clase para cargar y preparar datos de vibración desde archivos CSV.
    
    Esta clase maneja la carga de datos desde la estructura de carpetas
    donde cada carpeta representa un ensayo y contiene archivos CSV
    numerados consecutivamente que representan ventanas de análisis.
    """
    
    def __init__(self, base_path: str):
        """
        Inicializa el cargador de datos.
        
        Args:
            base_path (str): Ruta base donde se encuentran las carpetas de datos
        """
        self.base_path = base_path
        self.signal_processor = SignalProcessor()
        
    def load_experiment_data(self, 
                           folder_list: List[str], 
                           label: int,
                           preprocess: bool = True) -> Tuple[List[np.ndarray], List[int]]:
        """
        Carga datos de múltiples carpetas de experimentos.
        
        Args:
            folder_list (List[str]): Lista de carpetas a procesar
            label (int): Etiqueta para los datos (0: normal, 1: anormal)
            preprocess (bool): Si True, aplica preprocesamiento básico
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: Ventanas de datos y sus etiquetas
        """
        windows = []
        labels = []
        
        for folder in folder_list:
            folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(folder_path):
                print(f"Advertencia: La carpeta {folder_path} no existe")
                continue
                
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        # Cargar ventana
                        window = pd.read_csv(file_path, header=None).values
                        
                        if preprocess:
                            # Preprocesamiento básico
                            window = self.signal_processor.remove_dc(window)
                            window = self.signal_processor.normalize(window)
                            
                        windows.append(window)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error al cargar {file_path}: {str(e)}")
                        
        return windows, labels
    
    def prepare_dataset(self, 
                       normal_folders: List[str], 
                       anomaly_folders: List[str],
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Prepara el conjunto de datos completo para entrenamiento y prueba.
        
        Args:
            normal_folders (List[str]): Lista de carpetas con datos normales
            anomaly_folders (List[str]): Lista de carpetas con datos anómalos
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            Dict con conjuntos de entrenamiento y prueba
        """
        # Cargar datos normales y anómalos
        normal_windows, normal_labels = self.load_experiment_data(normal_folders, label=0)
        anomaly_windows, anomaly_labels = self.load_experiment_data(anomaly_folders, label=1)
        
        # Combinar datos
        all_windows = normal_windows + anomaly_windows
        all_labels = normal_labels + anomaly_labels
        
        # Convertir a arrays numpy
        X = np.array(all_windows)
        y = np.array(all_labels)
        
        # Aleatorizar los datos
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Dividir en entrenamiento y prueba
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    @staticmethod
    def get_experiment_info(folder_path: str) -> Dict[str, str]:
        """
        Extrae información del experimento desde el nombre de la carpeta.
        
        Args:
            folder_path (str): Ruta de la carpeta
            
        Returns:
            Dict[str, str]: Información del experimento
        """
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('_')
        
        info = {
            'rpm': parts[0] if len(parts) > 0 else 'unknown',
            'test_number': parts[1] if len(parts) > 1 else 'unknown'
        }
        
        return info
