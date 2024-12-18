# Conjunto de Datos de Ejemplo - Vibraciones en Motor de Inducción

Este directorio contiene un conjunto de datos de ejemplo para el análisis de vibraciones en un motor de inducción trifásico. Los datos fueron recolectados utilizando un acelerómetro MEMS ADXL345 de 3 ejes.

## Estructura del Conjunto de Datos

```
data/
├── normales/          # Ensayos en condiciones normales
│   ├── 1500_1/       # Ensayo a 1500 RPM, prueba 1
│   ├── 2000_1/       # Ensayo a 2000 RPM, prueba 1
│   └── 2500_1/       # Ensayo a 2500 RPM, prueba 1
├── anormales/        # Ensayos con condiciones anómalas
└── sin_correa/       # Ensayos específicos sin correa
```

## Características del Conjunto de Datos

### Formato de Archivos
- Cada archivo `.csv` contiene 780 muestras
- Tres columnas representando los ejes X, Y, Z del acelerómetro
- Sin encabezados en los archivos
- Nomenclatura incremental: `0000.csv`, `0001.csv`, `0002.csv`, etc.

### Estructura de Ensayos
- Cada subdirectorio representa un ensayo a una velocidad específica
- Nomenclatura: `[VELOCIDAD]_[NÚMERO_PRUEBA]`
  - Ejemplo: `2000_1` = Primera prueba a 2000 RPM

### Ciclo de Ensayo
Cada ensayo incluye:
1. Estado inicial (motor apagado)
2. Rampa de aceleración
3. Velocidad de régimen constante
4. Rampa de desaceleración
5. Estado final (motor apagado)

### Especificaciones Técnicas
- Sensor: ADXL345 (Acelerómetro MEMS triaxial)
- Frecuencia de muestreo: 1600 Hz
- Resolución: 13 bits
- Rango: ±16g

## Notas Importantes

1. **Chunks de Datos**
   - Cada archivo contiene exactamente 780 muestras
   - Los archivos son consecutivos en el tiempo
   - No hay solapamiento entre archivos

2. **Variabilidad de Ensayos**
   - Cada ensayo puede contener diferente cantidad de subidas/bajadas
   - La duración de las fases (aceleración, régimen, desaceleración) puede variar

3. **Uso Previsto**
   - Desarrollo y prueba de algoritmos
   - Validación de métodos de procesamiento
   - Ejemplos de implementación

## Limitaciones

Este es un conjunto de datos de ejemplo reducido, destinado principalmente para:
- Pruebas de funcionalidad
- Desarrollo de algoritmos
- Validación de métodos

Para aplicaciones que requieran un conjunto de datos más extenso, se recomienda esperar a la publicación del conjunto de datos completo en futuras versiones.

## Citación

Si utiliza este conjunto de datos en su investigación, por favor cite:
[Información de citación pendiente]
