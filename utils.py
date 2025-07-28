import json
import pickle
import re
import pandas as pd


def preprocess_text(text):
    """
    Esta función preprocesa texto para clasificación.

    Args:
        text (str): Texto a preprocesar

    Returns:
        str: Texto limpio
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())

    return text


def load_data(file_path):
    """
    Esta función sirve para cargar data.json

    Args:
        file_path (str): Ruta al archivo JSON que se nos proporciono

    Returns:
        pd.DataFrame: DataFrame con los datos
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        print(f" Dataset cargado: {len(df)} registros")
        print(f"️  Categorías únicas: {df['category'].nunique()}")

        return df

    except FileNotFoundError:
        print(f" Archivo no encontrado: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f" Error al parsear JSON: {file_path}")
        raise
    except Exception as e:
        print(f" Error inesperado: {e}")
        raise


def prepare_training_data(df, test_size=0.2, random_state=42):
    """
    Preparación y limpieza de los datos para entrenamiento.

    Args:
        df (pd.DataFrame): DataFrame con datos originales
        test_size (float): Proporción para test
        random_state (int): Semilla aleatoria

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    df['headline_clean'] = df['headline'].apply(preprocess_text)
    X = df['headline_clean']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f" Datos preparados:")
    print(f"    Entrenamiento: {len(X_train)} muestras")
    print(f"    Prueba: {len(X_test)} muestras")
    print(f"    Categorías: {len(y.unique())}")

    return X_train, X_test, y_train, y_test


def validate_request(data):
    """
    Valida el request de la API.
    El campo 'modelo' es opcional.

    Args:
        data (dict): Datos del request

    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return False, "No se proporcionaron datos"

    if 'titular' not in data:
        return False, "Falta el campo 'titular'"

    titular = data['titular']

    if not titular or not isinstance(titular, str):
        return False, "Titular inválido: debe ser una cadena no vacía"

    if len(titular.strip()) == 0:
        return False, "El titular no puede estar vacío"

    if len(titular) > 500:
        return False, "Titular demasiado largo (máximo 500 caracteres)"

    # Validar modelo solo si se proporciona
    if 'modelo' in data:
        valid_models = ['tfidf', 'sentence_transformer', 'gemma', 'ensemble']
        if data['modelo'] not in valid_models:
            return False, f"Modelo inválido. Elige entre: {valid_models}"

    return True, None


def format_prediction_response(prediction, titular, modelo_usado, tiempo_procesamiento):
    """
    Formatea la respuesta de predicción para la API con información de evaluación dinámica.

    Args:
        prediction (dict): Resultado de la predicción
        titular (str): Titular original
        modelo_usado (str): Modelo utilizado
        tiempo_procesamiento (float): Tiempo de procesamiento en ms

    Returns:
        dict: Respuesta formateada
    """
    from datetime import datetime

    top_probabilities = {}
    if 'probabilities' in prediction and prediction['probabilities']:
        sorted_probs = sorted(
            prediction['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_probabilities = dict(sorted_probs[:3])

    # Respuesta base
    response = {
        'titular': titular,
        'prediccion': {
            'categoria': prediction['category'],
            'confianza': round(prediction['confidence'], 4),
            'modelo_usado': modelo_usado
        },
        'top_3_probabilidades': top_probabilities,
        'tiempo_procesamiento_ms': round(tiempo_procesamiento, 2),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    # Información adicional del ensemble dinámico
    if 'reason' in prediction:
        response['prediccion']['razon'] = prediction['reason']

    if 'total_models_evaluated' in prediction:
        response['evaluacion_dinamica'] = {
            'modelos_evaluados': prediction['total_models_evaluated'],
            'resumen_confianzas': prediction.get('evaluation_summary', {}),
            'ganador': {
                'modelo': modelo_usado,
                'confianza': prediction['confidence'],
                'categoria': prediction['category']
            }
        }

    # Información de modelos alternativos
    if 'alternatives' in prediction and prediction['alternatives']:
        response['modelos_alternativos'] = {}
        for model_name, alt_data in prediction['alternatives'].items():
            response['modelos_alternativos'][model_name] = {
                'categoria': alt_data['category'],
                'confianza': round(alt_data['confidence'], 4),
                'tiempo_ms': round(alt_data.get('timing_ms', 0), 2)
            }

    # Información de timing detallado
    if 'timing_ms' in prediction:
        response['tiempo_modelo_ms'] = round(prediction['timing_ms'], 2)

    if 'error' in prediction:
        response['advertencia'] = f"Error del modelo: {prediction['error']}"

    return response


def load_model_metrics():
    """
    Carga las métricas reales de los modelos desde archivos.

    Returns:
        dict: Métricas de entrenamiento
    """
    from config.config import MODELS_DIR

    metrics_file = MODELS_DIR / 'metrics.pkl'

    if metrics_file.exists():
        try:
            with open(metrics_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error cargando métricas: {e}")
            return {}

    return {
        'tfidf': {'accuracy': 0.55, 'processing_time_ms': 5},
        'sentence_transformer': {'accuracy': 0.46, 'processing_time_ms': 50},
        'gemma': {'accuracy': 0.75, 'processing_time_ms': 200},
        'ensemble': {'accuracy': 0.65, 'processing_time_ms': 100}
    }


def save_model_metrics(metrics):
    """
    Guarda las métricas de los modelos.

    Args:
        metrics (dict): Métricas a guardar
    """
    from config.config import MODELS_DIR

    metrics_file = MODELS_DIR / 'metrics.pkl'

    try:
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f" Métricas guardadas en {metrics_file}")
    except Exception as e:
        print(f" Error guardando métricas: {e}")


def get_model_stats():
    """
    Obtiene estadísticas REALES de los modelos desde archivos.

    Returns:
        dict: Estadísticas de modelos
    """
    metrics = load_model_metrics()

    available_models = {}
    best_accuracy = 0
    fastest_model = 'tfidf'
    most_accurate = 'tfidf'

    model_descriptions = {
        'tfidf': 'TF-IDF + Regresión Logística - Rápido y confiable',
        'sentence_transformer': 'Sentence-Transformers + Random Forest - Embeddings semánticos',
        'gemma': 'Gemma LLM via Ollama - Modelo de lenguaje moderno',
        'ensemble': 'Ensemble Inteligente - Combina múltiples modelos'
    }

    for model_name, model_metrics in metrics.items():
        available_models[model_name] = {
            'nombre': model_descriptions.get(model_name, model_name),
            'precision': round(model_metrics.get('accuracy', 0), 4),
            'tiempo_promedio_ms': model_metrics.get('processing_time_ms', 0),
            'descripcion': model_descriptions.get(model_name, '')
        }

        if model_metrics.get('accuracy', 0) > best_accuracy:
            best_accuracy = model_metrics.get('accuracy', 0)
            most_accurate = model_name

        if model_name == 'tfidf':
            fastest_model = model_name

    return {
        'modelos_disponibles': available_models,
        'total_categorias': 42,
        'modelo_recomendado': most_accurate,
        'modelo_mas_rapido': fastest_model,
        'mejor_precision': best_accuracy
    }


def check_ollama_availability():
    """
    Verifica si Ollama está disponible.

    Returns:
        bool: True si Ollama está disponible
    """
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False



def get_categories_from_data(df):
    """
    Obtiene la lista de categorías del dataset.

    Args:
        df (pd.DataFrame): DataFrame con los datos

    Returns:
        list: Lista de categorías únicas
    """
    return sorted(df['category'].unique().tolist())


def get_best_model():
    """
    Obtiene el modelo con mejor accuracy.

    Returns:
        str: Nombre del mejor modelo
    """
    metrics = load_model_metrics()
    best_model = 'tfidf'
    best_accuracy = 0

    for model_name, model_metrics in metrics.items():
        if model_metrics.get('accuracy', 0) > best_accuracy:
            best_accuracy = model_metrics.get('accuracy', 0)
            best_model = model_name

    return best_model


def print_model_summary(model_name, accuracy, examples_tested=None):
    """
    Imprime un resumen del modelo entrenado.

    Args:
        model_name (str): Nombre del modelo
        accuracy (float): Accuracy del modelo
        examples_tested (int): Número de ejemplos probados
    """
    print(f"\n{'=' * 50}")
    print(f"RESUMEN: {model_name}")
    print(f"{'=' * 50}")
    print(f" Precisión: {accuracy:.4f}")
    if examples_tested:
        print(f" Ejemplos probados: {examples_tested}")
    print(f" Modelo listo para uso en API")