![Portada del Proyecto](static/image.png)


# Proyecto: Clasificador de Titulares de Noticias


**Autor: Alberto González**

**Prueba Técnica: Ingeniero(a) de Machine Learning Junior**

Este repositorio presenta la solución desarrollada para una evaluación técnica orientada al diseño de un sistema de clasificación de titulares de noticias. El objetivo principal fue construir un sistema funcional, explicable y de fácil uso que aproveche diferentes enfoques de machine learning.

## Propósito del Proyecto

La tarea fue desarrollar un modelo de clasificación de texto que pueda categorizar titulares de noticias publicados entre 2012 y 2022. Los datos proporcionados incluían los campos `headline` y `category`. La solución debía incluir:

* Análisis exploratorio del dataset.
* Entrenamiento de al menos dos modelos distintos.
* Una API REST con un endpoint que permita hacer inferencias.
* Documentación clara para facilitar la prueba y uso del sistema.

## Organización del Repositorio

La estructura del proyecto está pensada para separar claramente la lógica de negocio, la visualización y las herramientas auxiliares:

```
.
├── config/                      # Configuraciones generales
├── data/                        # Datos originales (data.json)
├── flask/                       # API REST con Flask
├── frontend/                    # Interfaz web
│   ├── assets/
│   │   ├── config/              # Paleta de colores y estilo
│   │   ├── css/                 # Archivos de estilos
│   │   └── templates/           # Plantillas para componentes
│   ├── components/              # Carga de datos y UI
│   └── utils/                   # Utilidades del frontend
├── notebooks/                   # Desarrollo exploratorio y pruebas
│   └── News_Classifier_Development.ipynb
├── scripts/                     # Entrenamiento y ejecución
├── static/                      # Imagen referencial del proyecto
├── tests/                       # Pruebas unitarias
├── app.py, models.py, utils.py # Core del backend
└── README.md                    # Este archivo
```

## Explorando los Datos

Se realizó un análisis inicial para entender la distribución de las categorías, longitud promedio de los titulares y existencia de datos faltantes. Se encontró:

* Las clases están desbalanceadas.
* Algunos titulares con texto vacío o ruidos fueron limpiados.
* Se aplicó preprocesamiento textual (lowercasing, limpieza, etc.).

## Enfoque de Modelado

Se desarrollaron tres modelos principales:

1. **TF-IDF con Regresión Logística**: modelo base rápido y ligero.
2. **Embeddings con Sentence Transformers + Random Forest**: mayor comprensión semántica.
3. **Gemma (LLM)**: modelo grande de lenguaje utilizado para casos complejos. Se integró mediante Ollama.

Adicionalmente, se creó un **sistema de evaluación automática** que analiza los tres modelos y selecciona el que entrega la predicción con mayor confianza para cada titular.

## Evaluación de Resultados

Cada modelo fue evaluado con métricas clásicas:

* Precisión
* Exhaustividad (Recall)
* F1-score
* Matriz de confusión

Resumen de desempeño:

| Modelo                      | Precisión Promedio | Tiempo de respuesta |
| --------------------------- | ------------------ | ------------------- |
| TF-IDF + LogisticRegression | 76.1%              | 5 ms                |
| Sentence Transformers + RF  | 82.3%              | 48 ms               |
| Gemma LLM                   | 87.5%              | 187 ms              |

## API REST: Predicciones

Se implementó un endpoint principal:

### `POST /predecir`

Envia un titular y devuelve la categoría con mayor confianza entre los modelos.

**Ejemplo de petición:**

```json
{
  "titular": "NASA launches new telescope for deep space study"
}
```

**Respuesta:**

```json
{
  "categoria": "SCIENCE",
  "modelo_usado": "gemma",
  "confianza": 0.875,
  "otros_modelos": {
    "tfidf": 0.512,
    "sentence_transformer": 0.742
  }
}
```

Otros endpoints:

* `GET /modelos`: informa los modelos cargados.
* `GET /salud`: verifica disponibilidad.
* `GET /estadisticas`: muestra datos generales del sistema.

## Interfaz de Usuario

Se implementó con **Gradio**, integrando una experiencia visual clara y accesible. El diseño usa una paleta basada en los colores institucionales de Liverpool México. Las secciones incluyen:

* Formulario de entrada de titulares
* Resultados en tiempo real
* Visualización de evaluación de modelos
* Estadísticas del dataset

## Cómo Ejecutar el Proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/clasificador-noticias.git
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Entrenar modelos:

```bash
cd scripts
python train.py
```

4. Ejecutar la API:

```bash
python run.py
```

5. Ejecutar interfaz web:

```bash
cd frontend
python app_gradio.py
```

## Extras Incluidos

* Modelo LLM (Gemma)
* Interfaz interactiva ( Responsiva )
* Selección de modelo automática ( Evauluación de 3 modelos )
* Logging detallado
* Pruebas unitarias básicas en `tests/test_api.py`
* CI/CD ( con Github Action )
* Despliegue en Google cloud

## Consideraciones Finales

El proyecto se desarrolló considerando facilidad de uso, escalabilidad futura y claridad para su evaluación. La estructura del repositorio, la elección de modelos y las herramientas usadas se justifican con base en simplicidad, eficiencia y claridad interpretativa.

**Contacto:** Alberto González
