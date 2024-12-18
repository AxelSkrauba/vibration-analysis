"""
Utilidades y funciones de visualización para análisis wavelet.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

def generate_test_signal(t: np.ndarray,
                        frequencies: List[float],
                        amplitudes: Optional[List[float]] = None,
                        noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una señal de prueba con múltiples componentes frecuenciales.
    
    Args:
        t: Vector de tiempo
        frequencies: Lista de frecuencias a incluir
        amplitudes: Lista de amplitudes para cada frecuencia
        noise_std: Desviación estándar del ruido gaussiano
        
    Returns:
        Tupla (señal_limpia, señal_con_ruido)
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    
    signal_clean = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        signal_clean += a * np.sin(2 * np.pi * f * t)
    
    noise = np.random.normal(0, noise_std, len(t))
    signal_noisy = signal_clean + noise
    
    return signal_clean, signal_noisy

def plot_signal_comparison(t: np.ndarray,
                         signals: Dict[str, np.ndarray],
                         title: str = "Comparación de Señales",
                         figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Compara múltiples señales en el mismo gráfico.
    
    Args:
        t: Vector de tiempo
        signals: Diccionario con nombres y señales
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    for name, signal in signals.items():
        plt.plot(t, signal, label=name)
    
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cwt_analysis(signal: np.ndarray, 
                     time: np.ndarray,
                     coefficients: np.ndarray,
                     scales: np.ndarray,
                     title: str = "Análisis CWT",
                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza el análisis CWT de una señal.
    
    Args:
        signal: Señal original
        time: Vector de tiempo
        coefficients: Coeficientes CWT
        scales: Escalas utilizadas
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title)
    
    # Señal original
    ax1.plot(time, signal)
    ax1.set_title("Señal Original")
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True)
    
    # Escalograma CWT
    im = ax2.pcolormesh(time, scales, np.abs(coefficients), 
                       shading='gouraud', cmap='viridis')
    ax2.set_title("Escalograma CWT")
    ax2.set_xlabel("Tiempo")
    ax2.set_ylabel("Escala")
    ax2.set_yscale('log')
    
    # Barra de color
    plt.colorbar(im, ax=ax2, label='Magnitud')
    plt.tight_layout()
    plt.show()

def plot_dwt_analysis(signal: np.ndarray,
                     time: np.ndarray,
                     approx: np.ndarray,
                     details: list,
                     title: str = "Análisis DWT",
                     figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Visualiza el análisis DWT de una señal.
    
    Args:
        signal: Señal original
        time: Vector de tiempo
        approx: Coeficientes de aproximación
        details: Lista de coeficientes de detalle
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    n_levels = len(details)
    fig, axes = plt.subplots(n_levels + 2, 1, figsize=figsize)
    fig.suptitle(title)
    
    # Señal original
    axes[0].plot(time, signal)
    axes[0].set_title("Señal Original")
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(True)
    
    # Aproximación
    t_approx = np.linspace(time[0], time[-1], len(approx))
    axes[1].plot(t_approx, approx)
    axes[1].set_title(f"Aproximación")
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel("Amplitud")
    axes[1].grid(True)
    
    # Detalles
    for i, detail in enumerate(details):
        t_detail = np.linspace(time[0], time[-1], len(detail))
        axes[i+2].plot(t_detail, detail)
        axes[i+2].set_title(f"Detalle nivel {i+1}")
        axes[i+2].set_xlabel("Tiempo")
        axes[i+2].set_ylabel("Amplitud")
        axes[i+2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_feature_comparison(features_dict: Dict[str, float],
                          title: str = "Comparación de Características",
                          figsize: Tuple[int, int] = (12, 6),
                          rotation: int = 45) -> None:
    """
    Visualiza la comparación de características extraídas.
    
    Args:
        features_dict: Diccionario de características
        title: Título del gráfico
        figsize: Tamaño de la figura
        rotation: Rotación de las etiquetas del eje x
    """
    plt.figure(figsize=figsize)
    
    # Preparar datos
    features = [(k, v) for k, v in features_dict.items() if not k.startswith('_')]
    names, values = zip(*features)
    
    # Crear gráfico de barras
    sns.barplot(x=list(names), y=list(values))
    plt.title(title)
    plt.xticks(rotation=rotation, ha='right')
    plt.xlabel("Características")
    plt.ylabel("Valor")
    
    plt.tight_layout()
    plt.show()

def compare_wavelet_features(signal: np.ndarray,
                           cwt_features: Dict[str, float],
                           dwt_features: Dict[str, float],
                           figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compara las características extraídas por CWT y DWT.
    
    Args:
        signal: Señal original
        cwt_features: Características extraídas por CWT
        dwt_features: Características extraídas por DWT
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Características CWT
    plt.subplot(2, 1, 1)
    plot_feature_comparison(cwt_features, "Características CWT")
    
    # Características DWT
    plt.subplot(2, 1, 2)
    plot_feature_comparison(dwt_features, "Características DWT")
    
    plt.tight_layout()
    plt.show()

def analyze_signal_components(signal: np.ndarray,
                            time: np.ndarray,
                            sampling_rate: float,
                            title: str = "Análisis de Componentes",
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Realiza un análisis completo de los componentes de la señal.
    
    Args:
        signal: Señal a analizar
        time: Vector de tiempo
        sampling_rate: Frecuencia de muestreo
        title: Título del análisis
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Señal en tiempo
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title(f"{title} - Dominio del Tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    
    # Espectro de frecuencia
    plt.subplot(2, 1, 2)
    frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
    spectrum = np.abs(np.fft.fft(signal))
    
    # Solo mostrar frecuencias positivas
    positive_freq_mask = frequencies >= 0
    plt.plot(frequencies[positive_freq_mask], spectrum[positive_freq_mask])
    plt.title("Espectro de Frecuencia")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
