from pathlib import Path
import os

# ==================== CARPETAS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== ARCHIVOS ====================
DATA_FILE = DATA_DIR / "data.json"
MODEL_FILES = {
    'tfidf_vectorizer': MODELS_DIR / 'tfidf_vectorizer.pkl',
    'tfidf_model': MODELS_DIR / 'tfidf_model.pkl',
    'sentence_transformer': MODELS_DIR / 'sentence_transformer.pkl'
}

# ==================== CONFIGURACION DE MI API ====================
API_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', 8000)),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true'
}

# ==================== CONFUGURACION DE MI ENTRENAMIENTO ====================
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# ==================== MODELO TF-IDF ====================
TFIDF_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'stop_words': 'english',
    'min_df': 2,
    'max_df': 0.95
}

# ==================== CATEGORÍAS DE NOTICIAS ====================
CATEGORIES = [
    'POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY',
    'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS',
    'COMEDY', 'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS',
    'THE WORLDPOST', 'WEDDINGS', 'WOMEN', 'CRIME', 'IMPACT', 'DIVORCE',
    'WORLD NEWS', 'MEDIA', 'WEIRD NEWS', 'GREEN', 'WORLDPOST', 'RELIGION',
    'STYLE', 'SCIENCE', 'TECH', 'TASTE', 'MONEY', 'ARTS', 'ENVIRONMENT',
    'FIFTY', 'GOOD NEWS', 'U.S. NEWS', 'ARTS & CULTURE', 'COLLEGE',
    'LATINO VOICES', 'CULTURE & ARTS', 'EDUCATION'
]

# ==================== OLLAMA SETTINGS ====================
OLLAMA_CONFIG = {
    'model': 'gemma:2b',
    'host': 'http://localhost:11434',
    'timeout': 30,
    'options': {
        'temperature': 0.1,
        'top_p': 0.9,
        'num_predict': 10
    }
}

# ==================== ENSEMBLE WEIGHTS ====================
ENSEMBLE_WEIGHTS = {
    'tfidf': 0.5484,           # Accuracy real
    'sentence_transformer': 0.4643,  # Accuracy real
    'gemma': 0.75              # Estimado (parece muy bueno en ejemplos)
}

# ==================== LOGGING ====================
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}