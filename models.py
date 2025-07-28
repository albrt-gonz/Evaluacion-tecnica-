import joblib
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

from utils import preprocess_text, load_model_metrics


class TfidfClassifier:
    """Clasificador usando TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False

    def train(self, X_train, y_train):
        """Entrena el modelo TF-IDF."""
        print("🚀 Entrenando modelo TF-IDF...")

        X_tfidf = self.vectorizer.fit_transform(X_train)

        self.model.fit(X_tfidf, y_train)
        self.is_trained = True

        print(" Modelo TF-IDF entrenado")

    def save(self, vectorizer_path, model_path):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.model, model_path)
        print(f" Modelo TF-IDF guardado en {model_path}")

    def load(self, vectorizer_path, model_path):
        """Carga modelo pre-entrenado."""
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(" Modelo TF-IDF cargado")

    def predict(self, X_test):
        """Realiza predicciones para múltiples muestras."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        X_tfidf = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)

        return predictions, probabilities

    def predict_single(self, headline):
        """Predice una sola muestra."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        clean_headline = preprocess_text(headline)
        X_tfidf = self.vectorizer.transform([clean_headline])
        prediction = self.model.predict(X_tfidf)[0]
        probabilities = self.model.predict_proba(X_tfidf)[0]

        # Crear diccionario de probabilidades
        prob_dict = dict(zip(self.model.classes_, probabilities))

        return {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {k: float(v) for k, v in prob_dict.items()},
            'model_name': 'tfidf'
        }


class SentenceTransformerClassifier:
    """Clasificador usando Sentence-Transformers + Random Forest."""

    def __init__(self):
        print(" Cargando Sentence-Transformer...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        print(" Sentence-Transformer cargado")

    def train(self, X_train, y_train):
        """Entrena el modelo con embeddings."""
        print(" Generando embeddings...")

        # Generar embeddings
        embeddings = self.encoder.encode(X_train.tolist(), show_progress_bar=True)

        # Entrenar clasificador
        print(" Entrenando Random Forest...")
        self.model.fit(embeddings, y_train)
        self.is_trained = True

        print(" Modelo Sentence-Transformer entrenado")

    def save(self, model_path):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        joblib.dump(self.model, model_path)
        print(f" Modelo Sentence-Transformer guardado en {model_path}")

    def load(self, model_path):
        """Carga modelo pre-entrenado."""
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(" Modelo Sentence-Transformer cargado")

    def predict(self, X_test):
        """Realiza predicciones para múltiples muestras."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        embeddings = self.encoder.encode(X_test.tolist(), show_progress_bar=True)
        predictions = self.model.predict(embeddings)
        probabilities = self.model.predict_proba(embeddings)

        return predictions, probabilities

    def predict_single(self, headline):
        """Predice una sola muestra."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        clean_headline = preprocess_text(headline)
        embedding = self.encoder.encode([clean_headline])
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]

        prob_dict = dict(zip(self.model.classes_, probabilities))

        return {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {k: float(v) for k, v in prob_dict.items()},
            'model_name': 'sentence_transformer'
        }


def _find_closest_category(prediction):
    """Encuentra la categoría más cercana."""
    prediction_lower = prediction.lower()

    if 'news' in prediction_lower:
        if 'u.s' in prediction_lower or 'us' in prediction_lower:
            return 'U.S. NEWS'
        else:
            return 'WORLD NEWS'
    elif 'comedy' in prediction_lower or 'funny' in prediction_lower:
        return 'COMEDY'
    elif 'parent' in prediction_lower:
        return 'PARENTING'
    elif 'sport' in prediction_lower:
        return 'SPORTS'
    elif 'business' in prediction_lower:
        return 'BUSINESS'
    elif 'politic' in prediction_lower:
        return 'POLITICS'
    elif 'wellness' in prediction_lower or 'health' in prediction_lower:
        return 'WELLNESS'
    elif 'entertainment' in prediction_lower:
        return 'ENTERTAINMENT'
    else:
        return 'POLITICS'  # Default a la categoría más común


class GemmaClassifier:
    """Clasificador usando Gemma LLM via Ollama."""

    def __init__(self, categories):
        self.categories = categories
        self.is_trained = True  # LLM no necesita entrenamiento
        print(" Gemma LLM inicializado")

    def create_prompt(self, headline):
        """Crea prompt para clasificación."""
        categories_str = ", ".join(self.categories)

        prompt = f"""Clasifica este titular de noticia en UNA de estas categorías: {categories_str}

Titular: "{headline}"

Responde SOLO con el nombre de la categoría de la lista anterior. Nada más.

Ejemplos:
- "Breaking: Major earthquake hits California" → U.S. NEWS
- "23 Funniest Tweets About Cats This Week" → COMEDY
- "Tips for helping toddler sleep better" → PARENTING

Categoría:"""

        return prompt

    def predict_single(self, headline):
        """Predice usando Gemma LLM."""
        try:
            prompt = self.create_prompt(headline)

            response = ollama.generate(
                model='gemma:2b',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 10
                }
            )

            predicted_category = response['response'].strip().upper()
            if predicted_category not in self.categories:
                predicted_category = _find_closest_category(predicted_category)

            return {
                'category': predicted_category,
                'confidence': 0.85,
                'probabilities': {predicted_category: 0.85},
                'model_name': 'gemma'
            }

        except Exception as e:
            print(f" Error con Gemma: {e}")
            return {
                'category': 'POLITICS',
                'confidence': 0.3,
                'probabilities': {'POLITICS': 0.3},
                'error': str(e),
                'model_name': 'gemma'
            }


class DynamicEnsembleClassifier:
    """Ensemble dinámico que evalúa los 3 modelos en tiempo real y selecciona el mejor."""

    def __init__(self, tfidf_model, st_model, gemma_model):
        self.tfidf_model = tfidf_model
        self.st_model = st_model
        self.gemma_model = gemma_model
        self.metrics = load_model_metrics()

        print(" Ensemble Dinámico inicializado - Evalúa los 3 modelos en tiempo real")

    def predict_single(self, headline):
        """
        Evalúa los 3 modelos disponibles y selecciona el de mayor confianza.
        """
        print(f"\n Evaluando titular: '{headline[:50]}...'")

        model_results = {}
        timing_info = {}

        print("   Evaluando con TF-IDF...")
        start_time = time.time()
        try:
            tfidf_result = self.tfidf_model.predict_single(headline)
            timing_info['tfidf'] = (time.time() - start_time) * 1000
            model_results['tfidf'] = tfidf_result
            print(f"     TF-IDF: {tfidf_result['category']} ({tfidf_result['confidence']:.3f})")
        except Exception as e:
            print(f"     TF-IDF falló: {e}")
            model_results['tfidf'] = None

        print("   Evaluando con Sentence-Transformer...")
        start_time = time.time()
        try:
            st_result = self.st_model.predict_single(headline)
            timing_info['sentence_transformer'] = (time.time() - start_time) * 1000
            model_results['sentence_transformer'] = st_result
            print(f"    ✅ ST: {st_result['category']} ({st_result['confidence']:.3f})")
        except Exception as e:
            print(f"    ❌ Sentence-Transformer falló: {e}")
            model_results['sentence_transformer'] = None

        if self.gemma_model:
            print("   Evaluando con Gemma LLM...")
            start_time = time.time()
            try:
                gemma_result = self.gemma_model.predict_single(headline)
                timing_info['gemma'] = (time.time() - start_time) * 1000
                model_results['gemma'] = gemma_result
                print(f"     Gemma: {gemma_result['category']} ({gemma_result['confidence']:.3f})")
            except Exception as e:
                print(f"     Gemma falló: {e}")
                model_results['gemma'] = None
        else:
            print("  ⚠  Gemma no disponible")
            model_results['gemma'] = None

        best_model = None
        best_result = None
        best_confidence = 0

        for model_name, result in model_results.items():
            if result and result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                best_model = model_name

        if not best_result:
            print("   Todos los modelos fallaron, usando fallback...")
            best_result = {
                'category': 'POLITICS',
                'confidence': 0.3,
                'probabilities': {'POLITICS': 0.3},
                'model_name': 'fallback'
            }
            best_model = 'fallback'

        alternatives = {}
        for model_name, result in model_results.items():
            if result and model_name != best_model:
                alternatives[model_name] = {
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'timing_ms': timing_info.get(model_name, 0)
                }

        final_result = {
            'category': best_result['category'],
            'confidence': best_result['confidence'],
            'probabilities': best_result.get('probabilities', {}),
            'model_used': best_model,
            'reason': f'mejor_confianza_{best_confidence:.3f}',
            'alternatives': alternatives,
            'timing_ms': timing_info.get(best_model, 0),
            'total_models_evaluated': len([r for r in model_results.values() if r]),
            'evaluation_summary': {
                'tfidf': model_results['tfidf']['confidence'] if model_results['tfidf'] else None,
                'sentence_transformer': model_results['sentence_transformer']['confidence'] if model_results[
                    'sentence_transformer'] else None,
                'gemma': model_results['gemma']['confidence'] if model_results['gemma'] else None
            }
        }

        print(f"   GANADOR: {best_model.upper()} con confianza {best_confidence:.3f}")
        print(f"   Modelos evaluados: {final_result['total_models_evaluated']}")

        return final_result

EnsembleClassifier = DynamicEnsembleClassifier