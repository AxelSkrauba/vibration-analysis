# Análisis de Vibraciones

Este proyecto implementa un sistema de análisis de señales de vibración para detección de anomalías en motores, utilizando datos de acelerómetros MEMS ADXL345 de 3 ejes. Se puede extender a otro tipo de acelerómetros, campos de aplicación y tipos de señales de vibración.

## Características Principales

- 🔍 Análisis multicanal de señales de vibración
- 📊 Extracción de características en dominios temporal y frecuencial
- 🤖 Detección de anomalías mediante modelos de una clase
- 📈 Visualización interactiva de distribuciones de frecuencia
- 🛠️ Preprocesamiento robusto de señales

## Estructura del Proyecto

```
vibration_analysis/
├── features/
│   ├── base.py              # Clases base para extractores
│   ├── time_domain.py       # Características temporales
│   └── frequency_domain.py  # Características frecuenciales
├── preprocessing/
│   └── signal_processing.py # Procesamiento de señales
└── utils/
    └── frequency_analysis_utils.py  # Utilidades de análisis frecuencial

examples/                # Notebooks de ejemplo
├── basic_usage.ipynb
├── frequency_distribution_analysis.ipynb
├── frequency_band_analysis.ipynb
└── anomaly_detection_oneclass.ipynb

data/                  # Datos de ejemplo
├── normales/          # Señales de funcionamiento normal
├── anormales/         # Señales de funcionamiento anómalo
└── sin_correa/        # Señales de funcionamiento anómalo específico
```

## Instalación
Se sugiere la utilización de un entorno virtual dedicado (virtualenv, por ejemplo)

```bash
git clone https://github.com/AxelSkrauba/vibration-analysis.git
cd vibration-analysis
```

Crear entorno virtual:
```bash
virtualenv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Instalar todo en el entorno virtual:
```bash
pip install -r requirements.txt
```

## Uso Rápido

```python
from vibration_analysis.preprocessing import SignalProcessor
from vibration_analysis.features.time_domain import TimeFeatureExtractor
from vibration_analysis.features.frequency_domain import FrequencyFeatureExtractor

# Preprocesamiento
processor = SignalProcessor()
signal_processed = processor.remove_dc(signal)
signal_processed = processor.normalize(signal_processed)

# Extracción de características
time_extractor = TimeFeatureExtractor()
freq_extractor = FrequencyFeatureExtractor(fs=1600)

# Obtener características
time_features = time_extractor.extract(signal_processed)
freq_features = freq_extractor.extract(signal_processed)
```

## Características Implementadas

### Dominio Temporal
- RMS (Root Mean Square)
- Valor Pico
- Factor de Cresta
- Kurtosis
- Asimetría (Skewness)
- Desviación Estándar
- Pico a Pico
- Factor de Impulso
- Factor de Forma

### Dominio Frecuencial
- Espectro de Potencia
- Frecuencias Dominantes
- Análisis de Bandas de Frecuencia
- Distribución Espectral
- Percentiles de Frecuencia

## Notebooks de Ejemplo
1. `basic_usage.ipynb`
   - Carga de datos
   - Extracción de características
   - Preparación de conjuntos de entrenamiento y prueba

2. `frequency_distribution_analysis.ipynb`
   - Análisis de distribución de frecuencias
   - Visualización de espectros
   - Detección de picos significativos

3. `frequency_band_analysis.ipynb`
   - Selección de bandas de frecuencia
   - Visualización de espectros
   - Detección de picos significativos

4. `anomaly_detection_oneclass.ipynb`
   - Entrenamiento de modelos de detección de anomalías
   - Comparación de diferentes algoritmos
   - Evaluación y visualización de resultados

## Roadmap 🚀
### Versión 0.1 (Actual)
- [x] Extracción de características temporales
- [x] Extracción de características frecuenciales
- [x] Análisis de distribución de frecuencias
- [x] Detección básica de anomalías
- [x] Visualización de resultados

### Próximas Versiones

#### Análisis Avanzado
- [x] Análisis Wavelet
- [ ] Análisis de Envolvente
- [ ] Análisis Tiempo-Frecuencia
- [ ] Demodulación de Señales
- [ ] Técnicas con Deep Learning

#### Machine Learning
- [ ] Clasificación Multiclase de Fallas
- [ ] Transfer Learning para Nuevos Tipos de Motores
- [ ] Métricas de Confianza en Predicciones

#### Optimización
- [ ] Procesamiento Paralelo
- [ ] Optimización de Memoria
- [ ] Compatibilidad con GPU
- [ ] Reducción de Dimensionalidad Automática

