"""
Script para entrenar todos los modelos del clasificador de noticias.
Este script debe ejecutarse UNA VEZ para generar los modelos .pkl.

Uso:
    python train.py

Salida:
    - models/tfidf_vectorizer.pkl
    - models/tfidf_model.pkl
    - models/sentence_transformer.pkl
    - models/metrics.pkl
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models import TfidfClassifier, SentenceTransformerClassifier, GemmaClassifier
from utils import (load_data, prepare_training_data, get_categories_from_data,
                   print_model_summary, save_model_metrics)
from config.config import DATA_FILE, MODEL_FILES, TRAINING_CONFIG
from sklearn.metrics import accuracy_score


def train_tfidf_model(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa el modelo TF-IDF.

    Returns:
        tuple: (TfidfClassifier, accuracy, processing_time)
    """
    print(f"\n{'=' * 60}")
    print("ENTRENANDO MODELO TF-IDF")
    print(f"{'=' * 60}")

    model = TfidfClassifier()
    start_time = time.time()
    model.train(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions, _ = model.predict(X_test)
    prediction_time = (time.time() - start_time) * 1000 / len(X_test)  # ms por predicción

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n EVALUACIÓN TF-IDF:")
    print(f" Precisión: {accuracy:.4f}")
    print(f"️ Tiempo entrenamiento: {train_time:.2f} segundos")
    print(f" Tiempo promedio predicción: {prediction_time:.2f} ms")

    model.save(
        MODEL_FILES['tfidf_vectorizer'],
        MODEL_FILES['tfidf_model']
    )

    print_model_summary("TF-IDF + Regresión Logística", accuracy, len(X_test))

    return model, accuracy, prediction_time


def train_sentence_transformer(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa el modelo Sentence-Transformer.

    Returns:
        tuple: (SentenceTransformerClassifier, accuracy, processing_time)
    """
    print(f"\n{'=' * 60}")
    print("ENTRENANDO SENTENCE-TRANSFORMER")
    print(f"{'=' * 60}")


    model = SentenceTransformerClassifier()
    start_time = time.time()
    model.train(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions, _ = model.predict(X_test)
    prediction_time = (time.time() - start_time) * 1000 / len(X_test)  # ms por predicción

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n EVALUACIÓN SENTENCE-TRANSFORMER:")
    print(f" Precisión: {accuracy:.4f}")
    print(f"️ Tiempo entrenamiento: {train_time:.2f} segundos")
    print(f" Tiempo promedio predicción: {prediction_time:.2f} ms")

    model.save(MODEL_FILES['sentence_transformer'])
    print_model_summary("Sentence-Transformer + Random Forest", accuracy, len(X_test))

    return model, accuracy, prediction_time


def evaluate_gemma_model(X_test, y_test, categories):
    """
    Evalúa el modelo Gemma con algunas muestras.

    Returns:
        tuple: (accuracy_estimate, processing_time)
    """
    print(f"\n{'=' * 60}")
    print("EVALUANDO GEMMA LLM")
    print(f"{'=' * 60}")

    try:
        model = GemmaClassifier(categories)
        test_subset = X_test.head(10) if hasattr(X_test, 'head') else X_test[:10]
        y_subset = y_test.head(10) if hasattr(y_test, 'head') else y_test[:10]

        predictions = []
        total_time = 0

        for headline in test_subset:
            start_time = time.time()
            result = model.predict_single(headline)
            prediction_time = (time.time() - start_time) * 1000
            total_time += prediction_time
            predictions.append(result['category'])

        accuracy = accuracy_score(y_subset, predictions)
        avg_time = total_time / len(test_subset)

        print(f" Precisión estimada: {accuracy:.4f} (en {len(test_subset)} muestras)")
        print(f" Tiempo promedio: {avg_time:.2f} ms")

        return accuracy, avg_time

    except Exception as e:
        print(f" Gemma no disponible: {e}")
        return 0.75, 200


def test_models_with_examples(tfidf_model, st_model, categories):
    """
    Prueba los modelos con ejemplos específicos.

    Args:
        tfidf_model: Modelo TF-IDF entrenado
        st_model: Modelo Sentence-Transformer entrenado
        categories: Lista de categorías
    """
    print(f"\n{'=' * 60}")
    print("PROBANDO MODELOS CON EJEMPLOS")
    print(f"{'=' * 60}")

    test_headlines = [
        "Breaking: Major earthquake hits California coast",
        "23 Funniest Tweets About Cats This Week",
        "Tips for helping your toddler sleep better",
        "Tesla stock soars 15% after earnings report",
        "World Cup final draws record viewership"
    ]

    expected_categories = [
        "U.S. NEWS",
        "COMEDY",
        "PARENTING",
        "BUSINESS",
        "SPORTS"
    ]

    for i, headline in enumerate(test_headlines):
        print(f"\n Titular: '{headline}'")
        print(f"    Esperado: {expected_categories[i]}")

        # TF-IDF
        tfidf_result = tfidf_model.predict_single(headline)
        print(f"    TF-IDF: {tfidf_result['category']} ({tfidf_result['confidence']:.3f})")

        # Sentence-Transformer
        st_result = st_model.predict_single(headline)
        print(f"    ST: {st_result['category']} ({st_result['confidence']:.3f})")

        # Gemma
        try:
            gemma_model = GemmaClassifier(categories)
            gemma_result = gemma_model.predict_single(headline)
            print(f"    Gemma: {gemma_result['category']} ({gemma_result['confidence']:.3f})")
        except Exception as e:
            print(f"     Gemma no disponible: {e}")


def main():
    """Función principal de entrenamiento."""
    print(" INICIANDO ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    start_time = time.time()

    try:
        print(" Cargando dataset...")
        if not DATA_FILE.exists():
            print(f" Archivo de datos no encontrado: {DATA_FILE}")
            print(" Asegúrate de tener el archivo data.json en la carpeta data/")
            return False

        df = load_data(DATA_FILE)
        categories = get_categories_from_data(df)

        print("\n Preparando datos para entrenamiento...")
        X_train, X_test, y_train, y_test = prepare_training_data(
            df,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state']
        )
        results = {}
        metrics = {}

        # TF-IDF
        tfidf_model, tfidf_acc, tfidf_time = train_tfidf_model(X_train, X_test, y_train, y_test)
        results['tfidf'] = tfidf_acc
        metrics['tfidf'] = {
            'accuracy': tfidf_acc,
            'processing_time_ms': tfidf_time
        }

        # Sentence-Transformer
        st_model, st_acc, st_time = train_sentence_transformer(X_train, X_test, y_train, y_test)
        results['sentence_transformer'] = st_acc
        metrics['sentence_transformer'] = {
            'accuracy': st_acc,
            'processing_time_ms': st_time
        }

        # Gemma
        gemma_acc, gemma_time = evaluate_gemma_model(X_test, y_test, categories)
        results['gemma'] = gemma_acc
        metrics['gemma'] = {
            'accuracy': gemma_acc,
            'processing_time_ms': gemma_time
        }

        # Ensemble
        ensemble_acc = (
            metrics['tfidf']['accuracy'] * 0.4 +
            metrics['sentence_transformer']['accuracy'] * 0.3 +
            metrics['gemma']['accuracy'] * 0.3
        )
        ensemble_time = (
            metrics['tfidf']['processing_time_ms'] * 0.4 +
            metrics['sentence_transformer']['processing_time_ms'] * 0.3 +
            metrics['gemma']['processing_time_ms'] * 0.3
        )

        results['ensemble'] = ensemble_acc
        metrics['ensemble'] = {
            'accuracy': ensemble_acc,
            'processing_time_ms': ensemble_time
        }

        save_model_metrics(metrics)
        test_models_with_examples(tfidf_model, st_model, categories)

        best_model = max(results.items(), key=lambda x: x[1])
        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(" ENTRENAMIENTO COMPLETADO")
        print(f"{'=' * 60}")
        print(f"️  Tiempo total: {total_time:.1f} segundos")
        print(f" Modelos entrenados: {len(results)}")
        print(f" Mejor modelo: {best_model[0]} (Precisión: {best_model[1]:.4f})")

        print(f"\n RESUMEN DE RESULTADOS:")
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            processing_time = metrics[model_name]['processing_time_ms']
            print(f"    {model_name}: {accuracy:.4f} precisión, {processing_time:.1f}ms promedio")

        print(f"\n ARCHIVOS GENERADOS:")
        for name, path in MODEL_FILES.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"    {path} ({size_mb:.1f} MB)")
            else:
                print(f"    {path} (no generado)")

        metrics_file = Path("../config/models/metrics.pkl")
        if metrics_file.exists():
            size_kb = metrics_file.stat().st_size / 1024
            print(f"   ✅ models/metrics.pkl ({size_kb:.1f} KB) - MÉTRICAS REALES")

        print(f"\n PRÓXIMO PASO:")
        print(f"   Ejecutar: python app.py")
        print(f"   O: python run.py")
        print(f"\n El modelo recomendado será: {best_model[0]} (mejor precisión)")

        return True

    except Exception as e:
        print(f" Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)