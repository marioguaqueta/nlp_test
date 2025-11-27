# Proyecto de Canonicalización de Pedidos con Qwen/Qwen3-0.6B-Base

Este proyecto tiene como objetivo afinar (fine-tune) el modelo `Qwen/Qwen3-0.6B-Base` para convertir órdenes de compra en lenguaje natural a un formato JSON estructurado específico.

## Estructura del Proyecto

- `data/`: Coloca aquí los archivos `train.csv` y `test.csv`.
- `src/`: Código fuente del proyecto.
    - `data_loader.py`: Procesamiento de datos.
    - `train.py`: Script de entrenamiento.
    - `inference.py`: Generación de predicciones.
    - `metrics.py`: Implementación de la métrica de evaluación personalizada.
- `models/`: Directorio donde se guardarán los modelos entrenados.
- `output/`: Directorio para el archivo `submission.csv`.
- `notebooks/`: Notebooks para exploración y pruebas.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. **Preparar datos**: Asegúrate de que `train.csv` y `test.csv` estén en la carpeta `data/`.
2. **Entrenar**:
   ```bash
   python src/train.py
   ```
3. **Generar Predicciones**:
   ```bash
   python src/inference.py
   ```
