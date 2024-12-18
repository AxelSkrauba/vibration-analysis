import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import glob
from typing import Optional, Tuple, List

def load_signal(file_path):
    """Carga una señal desde un archivo CSV."""
    return pd.read_csv(file_path, header=None).values

def plot_spectrum(signal: np.ndarray, fs: float, low_freq_cutoff: Optional[float] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Grafica el espectro de frecuencia de la señal.
    
    Args:
        signal: Señal de entrada
        fs: Frecuencia de muestreo en Hz
        low_freq_cutoff: Frecuencia de corte inferior. Frecuencias por debajo serán omitidas del gráfico
    """
    n_channels = signal.shape[1] if len(signal.shape) > 1 else 1
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 4*n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        channel_signal = signal[:, i] if n_channels > 1 else signal
        fft = fftpack.fft(channel_signal)
        freqs = fftpack.fftfreq(len(channel_signal), d=1/fs)[:len(channel_signal)//2]
        magnitude = np.abs(fft)[:len(channel_signal)//2]
        
        if low_freq_cutoff is not None:
            # Crear máscara para frecuencias bajas
            mask = freqs >= low_freq_cutoff
            plot_freqs = freqs[mask]
            plot_magnitude = magnitude[mask]
        else:
            plot_freqs = freqs
            plot_magnitude = magnitude
        
        axes[i].plot(plot_freqs, plot_magnitude)
        axes[i].set_xlabel('Frecuencia (Hz)')
        axes[i].set_ylabel('Magnitud')
        axes[i].set_title(f'Espectro Canal {i+1}')
        axes[i].grid(True)
    
    plt.tight_layout()
    return fig, axes

def highlight_band(ax, freq_min, freq_max, color='red', alpha=0.2):
    """Resalta una banda de frecuencia en el gráfico."""
    ylims = ax.get_ylim()
    ax.fill_between([freq_min, freq_max], [ylims[0], ylims[0]], 
                    [ylims[1], ylims[1]], color=color, alpha=alpha)

class FrequencyBandAnalyzer:
    def __init__(self, fs: Optional[float] = None, freq_min: Optional[float] = None, 
                 freq_max: Optional[float] = None, low_freq_cutoff: Optional[float] = None):
        """
        Inicializa el analizador de bandas de frecuencia.
        
        Args:
            fs: Frecuencia de muestreo en Hz. Si no se especifica, se determinará de la señal
            freq_min: Frecuencia mínima para los sliders. Si no se especifica, será 0
            freq_max: Frecuencia máxima para los sliders. Si no se especifica, será fs/2
            low_freq_cutoff: Frecuencia de corte inferior. Frecuencias por debajo serán omitidas
        """
        self.fs = fs
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.low_freq_cutoff = low_freq_cutoff
        self.bands_list = []
        self.example_signal = None
        self.output = widgets.Output()
        
        # Los sliders se crearán cuando se establezca la señal
        self.band_min = None
        self.band_max = None
        self.add_band_btn = widgets.Button(description='Agregar Banda')
        self.clear_bands_btn = widgets.Button(description='Limpiar Bandas')
        self.low_freq_toggle = widgets.Checkbox(
            value=low_freq_cutoff is not None,
            description='Omitir bajas frecuencias',
            style={'description_width': 'initial'}
        )
        self.low_freq_slider = widgets.FloatSlider(
            value=low_freq_cutoff if low_freq_cutoff is not None else 0,
            min=0,
            max=10,
            step=0.1,
            description='Corte inferior (Hz)',
            disabled=not self.low_freq_toggle.value,
            style={'description_width': 'initial'}
        )
        
        # Conectar callbacks
        self.add_band_btn.on_click(self.add_band)
        self.clear_bands_btn.on_click(self.clear_bands)
        self.low_freq_toggle.observe(self._on_low_freq_toggle_change, names='value')
        self.low_freq_slider.observe(self._on_low_freq_slider_change, names='value')
    
    def _create_sliders(self):
        """Crea los sliders basados en la frecuencia de muestreo."""
        if self.fs is None:
            raise ValueError("Frecuencia de muestreo no establecida")
            
        nyquist = self.fs / 2
        min_val = self.freq_min if self.freq_min is not None else 0
        max_val = self.freq_max if self.freq_max is not None else nyquist
        
        self.band_min = widgets.FloatSlider(
            value=min_val,
            min=min_val,
            max=max_val,
            step=0.1,
            description='Freq Min (Hz)',
            style={'description_width': 'initial'}
        )
        
        self.band_max = widgets.FloatSlider(
            value=min_val + (max_val - min_val) / 4,  # 25% del rango por defecto
            min=min_val,
            max=max_val,
            step=0.1,
            description='Freq Max (Hz)',
            style={'description_width': 'initial'}
        )
        
        # Actualizar límites del slider de corte de bajas frecuencias
        self.low_freq_slider.max = max_val / 5  # 20% de la frecuencia máxima
    
    def _on_low_freq_toggle_change(self, change):
        """Maneja cambios en el checkbox de omitir bajas frecuencias."""
        self.low_freq_slider.disabled = not change['new']
        self.low_freq_cutoff = self.low_freq_slider.value if change['new'] else None
        self.update_plot()
    
    def _on_low_freq_slider_change(self, change):
        """Maneja cambios en el slider de corte de bajas frecuencias."""
        if self.low_freq_toggle.value:
            self.low_freq_cutoff = change['new']
            self.update_plot()
    
    def set_signal(self, signal: np.ndarray, fs: Optional[float] = None):
        """
        Establece la señal a analizar.
        
        Args:
            signal: Señal de entrada
            fs: Frecuencia de muestreo en Hz. Si no se especifica, se usa la establecida en __init__
        """
        self.example_signal = signal
        if fs is not None:
            self.fs = fs
        elif self.fs is None:
            raise ValueError("Debe especificar la frecuencia de muestreo")
            
        self._create_sliders()
        self.update_plot()
    
    def update_plot(self):
        """Actualiza el gráfico con las bandas seleccionadas."""
        if self.example_signal is None:
            return
            
        with self.output:
            clear_output(wait=True)
            fig, axes = plot_spectrum(self.example_signal, self.fs, self.low_freq_cutoff)
            
            # Asegurarnos de que axes sea una lista
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            elif isinstance(axes, np.ndarray):
                axes = axes.flatten()
            
            # Agregar todas las bandas a cada eje
            for ax in axes:
                for band in self.bands_list:
                    highlight_band(ax, band[0], band[1])
            
            plt.show()
    
    def add_band(self, b):
        """Agrega una nueva banda de frecuencia."""
        min_freq = self.band_min.value
        max_freq = self.band_max.value
        if min_freq < max_freq:
            self.bands_list.append((min_freq, max_freq))
            self.update_plot()
            print(f'Banda agregada: {min_freq:.1f}-{max_freq:.1f} Hz')
            # Mostrar todas las bandas actuales
            print("\nBandas actuales:")
            for i, (fmin, fmax) in enumerate(self.bands_list, 1):
                print(f"Banda {i}: {fmin:.1f}-{fmax:.1f} Hz")
    
    def clear_bands(self, b):
        """Limpia todas las bandas."""
        self.bands_list.clear()
        self.update_plot()
        print('Bandas limpiadas')
    
    def display_controls(self):
        """Muestra los controles interactivos."""
        if self.band_min is None or self.band_max is None:
            raise ValueError("Debe establecer una señal primero usando set_signal()")
            
        display(widgets.VBox([
            widgets.HBox([self.low_freq_toggle, self.low_freq_slider]),
            self.band_min,
            self.band_max,
            widgets.HBox([self.add_band_btn, self.clear_bands_btn]),
            self.output
        ]))
    
    def get_bands(self):
        """Retorna la lista de bandas seleccionadas."""
        return self.bands_list

class FrequencyDistributionAnalyzer:
    """Clase para analizar la distribución de frecuencias de señales."""
    
    def __init__(self, fs=1000.0):
        """
        Inicializa el analizador.
        
        Args:
            fs (float): Frecuencia de muestreo en Hz
        """
        self.fs = fs
        self.signals = []  # Lista de señales por canal
        self.n_channels = 0  # Número de canales
        self.current_channel = 0  # Canal actual para análisis
        
    def load_signals_from_directory(self, directory_path, channel=None):
        """
        Carga señales desde un directorio, organizándolas por canal.
        
        Args:
            directory_path (str): Ruta al directorio con los archivos de señales
            channel (int, optional): Canal específico a cargar. Si es None, carga todos los canales.
        """
        self.signals = []  # Reset signals list
        
        # Obtener lista de archivos .txt o .dat
        files = glob.glob(os.path.join(directory_path, "*.csv"))
                
        if not files:
            raise ValueError(f"No se encontraron archivos .txt o .dat en {directory_path}")
            
        # Cargar primera señal para determinar número de canales
        first_signal = load_signal(files[0])
        if len(first_signal.shape) == 1:
            self.n_channels = 1
        else:
            self.n_channels = first_signal.shape[1]
            
        # Inicializar listas para cada canal
        self.signals = [[] for _ in range(self.n_channels)]
        
        # Cargar señales por canal
        for file_path in files:
            signal_data = load_signal(file_path)
            
            # Convertir a 2D si es necesario
            if len(signal_data.shape) == 1:
                signal_data = signal_data.reshape(-1, 1)
                
            # Almacenar cada canal por separado
            for ch in range(self.n_channels):
                if channel is None or channel == ch:
                    self.signals[ch].append(signal_data[:, ch])
                    
        # Si se especificó un canal, mantener solo ese canal
        if channel is not None:
            self.signals = [self.signals[channel]]
            self.n_channels = 1
            self.current_channel = 0
        else:
            self.current_channel = 0
            
        print(f"Se cargaron {len(self.signals[0])} señales con {self.n_channels} canales")
        
    def set_current_channel(self, channel):
        """
        Establece el canal actual para análisis.
        
        Args:
            channel (int): Número de canal a analizar
        """
        if channel < 0 or channel >= self.n_channels:
            raise ValueError(f"Canal {channel} inválido. Debe estar entre 0 y {self.n_channels-1}")
        self.current_channel = channel
        
    def plot_fft_distribution(self, freq_range=None, normalize=False, yscale='log'):
        """
        Grafica la distribución de FFTs para el canal actual.
        
        Args:
            freq_range (tuple): Rango de frecuencias a mostrar (min_freq, max_freq)
            normalize (bool): Si es True, normaliza las FFTs
            yscale (str): Escala del eje y ('log' o 'linear')
        """
        if not self.signals or not self.signals[self.current_channel]:
            raise ValueError("No hay señales cargadas")
            
        plt.figure(figsize=(12, 6))
        
        # Calcular FFTs para el canal actual
        ffts = []
        for signal in self.signals[self.current_channel]:
            fft_result = np.abs(np.fft.rfft(signal))
            if normalize:
                fft_result = fft_result / np.max(fft_result)
            ffts.append(fft_result)
            
        ffts = np.array(ffts)
        freqs = np.fft.rfftfreq(len(self.signals[self.current_channel][0]), d=1/self.fs)
        
        # Calcular percentiles
        p25 = np.percentile(ffts, 25, axis=0)
        p50 = np.percentile(ffts, 50, axis=0)
        p75 = np.percentile(ffts, 75, axis=0)
        
        if freq_range:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs = freqs[mask]
            p25 = p25[mask]
            p50 = p50[mask]
            p75 = p75[mask]
            
        plt.fill_between(freqs, p25, p75, alpha=0.3, label='25-75 percentil')
        plt.plot(freqs, p50, 'r-', label='Mediana', linewidth=2)
        
        plt.grid(True)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud' + (' Normalizada' if normalize else ''))
        plt.yscale(yscale)
        plt.title(f'Distribución de FFT - Canal {self.current_channel}')
        plt.legend()
        plt.tight_layout()
        
    def detect_peaks(self, prominence=None, width=None):
        """
        Detecta picos significativos en el espectro promedio.
        
        Args:
            prominence: Prominencia mínima para considerar un pico
            width: Ancho mínimo del pico
        """
        from scipy.signal import find_peaks
        
        # Calcular la FFT promedio
        signals_array = np.array(self.signals[self.current_channel])
        n_samples = signals_array.shape[1]
        n_fft = n_samples // 2 + 1  # Longitud de la FFT real
        
        # Calcular FFT para cada señal y promediar
        ffts = np.abs(np.fft.rfft(signals_array, axis=1))
        mean_fft = np.mean(ffts, axis=0)
        
        # Encontrar picos
        peaks, properties = find_peaks(mean_fft, prominence=prominence, width=width)
        
        # Calcular frecuencias correspondientes a los picos
        freqs = np.fft.rfftfreq(n_samples, d=1/self.fs)
        
        return [(freqs[peak], mean_fft[peak]) for peak in peaks]
        
    def compare_groups(self, other_analyzer, title="Group Comparison"):
        """
        Compara las distribuciones de FFT entre dos grupos de señales.
        
        Args:
            other_analyzer: Otro FrequencyDistributionAnalyzer para comparar
            title: Título del gráfico
        """
        # Calcular FFTs para el primer grupo
        signals_array1 = np.array(self.signals[self.current_channel])
        n_samples1 = signals_array1.shape[1]
        ffts_array1 = np.abs(np.fft.rfft(signals_array1, axis=1))
        mean_fft1 = np.mean(ffts_array1, axis=0)
        std_fft1 = np.std(ffts_array1, axis=0)
        freqs1 = np.fft.rfftfreq(n_samples1, d=1/self.fs)

        # Calcular FFTs para el segundo grupo
        signals_array2 = np.array(other_analyzer.signals[other_analyzer.current_channel])
        n_samples2 = signals_array2.shape[1]
        ffts_array2 = np.abs(np.fft.rfft(signals_array2, axis=1))
        mean_fft2 = np.mean(ffts_array2, axis=0)
        std_fft2 = np.std(ffts_array2, axis=0)
        freqs2 = np.fft.rfftfreq(n_samples2, d=1/other_analyzer.fs)

        plt.figure(figsize=(12, 6))
        
        # Graficar primer grupo
        plt.plot(freqs1, mean_fft1, 'b-', label='Group 1 Mean', alpha=0.8)
        plt.fill_between(freqs1, mean_fft1-std_fft1, mean_fft1+std_fft1, 
                        color='blue', alpha=0.2)
        
        # Graficar segundo grupo
        plt.plot(freqs2, mean_fft2, 'r-', label='Group 2 Mean', alpha=0.8)
        plt.fill_between(freqs2, mean_fft2-std_fft2, mean_fft2+std_fft2,
                        color='red', alpha=0.2)
        
        plt.title(title)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
    def create_interactive_plot(self):
        """Crea una interfaz interactiva para explorar las distribuciones de FFT."""
        freq_min = widgets.FloatSlider(
            value=0,
            min=0,
            max=self.fs/2,
            step=1,
            description='Freq Min (Hz)',
            style={'description_width': 'initial'}
        )
        
        freq_max = widgets.FloatSlider(
            value=self.fs/2,
            min=0,
            max=self.fs/2,
            step=1,
            description='Freq Max (Hz)',
            style={'description_width': 'initial'}
        )
        
        normalize = widgets.Checkbox(
            value=False,
            description='Normalize FFTs',
            style={'description_width': 'initial'}
        )
        
        yscale = widgets.Dropdown(
            options=['log', 'linear'],
            value='log',
            description='Y Scale',
            style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        def update_plot(change=None):
            with output:
                output.clear_output(wait=True)
                self.plot_fft_distribution(
                    freq_range=(freq_min.value, freq_max.value),
                    normalize=normalize.value,
                    yscale=yscale.value
                )
                plt.show()
        
        # Conectar los widgets a la función de actualización
        freq_min.observe(update_plot, names='value')
        freq_max.observe(update_plot, names='value')
        normalize.observe(update_plot, names='value')
        yscale.observe(update_plot, names='value')
        
        # Crear el layout de los controles
        controls = widgets.VBox([
            widgets.HTML("<h3>FFT Distribution Controls</h3>"),
            freq_min,
            freq_max,
            normalize,
            yscale
        ])
        
        # Layout principal
        layout = widgets.HBox([controls, output])
        
        # Mostrar el plot inicial
        update_plot()
        
        return layout
