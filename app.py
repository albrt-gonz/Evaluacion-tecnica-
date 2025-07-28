from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models import TfidfClassifier, SentenceTransformerClassifier, GemmaClassifier, EnsembleClassifier
from utils import (validate_request, format_prediction_response, get_model_stats,
                   check_ollama_availability, get_best_model)
from config.config import MODELS_DIR, API_CONFIG, CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClasificadorNoticiasAPI:
    """Mi clase prinicpal donde arranco la aplicación."""

    def __init__(self):
        self.models = {}
        self.ollama_available = False
        self.categories = CATEGORIES
        self.load_models()

    def load_models(self):
        """Carga todos los modelos pre-entrenados."""
        logger.info("Cargando modelos...")

        try:
            self.models['tfidf'] = TfidfClassifier()
            self.models['tfidf'].load(
                MODELS_DIR / 'tfidf_vectorizer.pkl',
                MODELS_DIR / 'tfidf_model.pkl'
            )
            self.models['sentence_transformer'] = SentenceTransformerClassifier()
            self.models['sentence_transformer'].load(MODELS_DIR / 'sentence_transformer.pkl')
            self.ollama_available = check_ollama_availability()

            if self.ollama_available:
                self.models['gemma'] = GemmaClassifier(self.categories)
                logger.info("Gemma LLM disponible")
            else:
                logger.warning("Ollama no disponible - Gemma deshabilitado")

            if self.ollama_available:
                self.models['ensemble'] = EnsembleClassifier(
                    self.models['tfidf'],
                    self.models['sentence_transformer'],
                    self.models['gemma']
                )
            else:
                self.models['ensemble'] = EnsembleClassifier(
                    self.models['tfidf'],
                    self.models['sentence_transformer'],
                    None
                )

            logger.info(f"{len(self.models)} modelos cargados exitosamente")

        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            raise

    def predecir(self, titular, nombre_modelo='ensemble'):
        """
        Realiza predicción usando el modelo especificado.

        Args:
            titular (str): Titular a clasificar
            nombre_modelo (str): Nombre del modelo a usar

        Returns:
            dict: Resultado de la predicción
        """
        if nombre_modelo not in self.models:
            modelos_disponibles = list(self.models.keys())
            raise ValueError(f"Modelo '{nombre_modelo}' no disponible. Disponibles: {modelos_disponibles}")

        if nombre_modelo == 'gemma' and not self.ollama_available:
            raise ValueError("Gemma no está disponible (Ollama no configurado)")

        start_time = time.time()

        try:
            result = self.models[nombre_modelo].predict_single(titular)
            tiempo_procesamiento = (time.time() - start_time) * 1000

            return format_prediction_response(result, titular, nombre_modelo, tiempo_procesamiento)

        except Exception as e:
            logger.error(f"Error en predicción con {nombre_modelo}: {e}")
            raise

try:
    clasificador_api = ClasificadorNoticiasAPI()
    logger.info("API inicializada correctamente")
except Exception as e:
    logger.error(f"Error inicializando API: {e}")
    clasificador_api = None


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint no encontrado',
            'endpoints_disponibles': [
                'POST /predecir',
                'GET /modelos',
                'GET /salud',
                'GET /estadisticas'
            ]
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Error interno del servidor',
            'mensaje': 'Por favor contacta al administrador'
        }), 500

    @app.route('/', methods=['GET'])
    def inicio():
        """Endpoint de inicio."""
        return jsonify({
            'mensaje': '📰 API Clasificador de Noticias',
            'version': '1.0.0',
            'estado': 'activo',
            'endpoints': {
                'predecir': 'POST /predecir - Clasificar titular',
                'modelos': 'GET /modelos - Modelos disponibles',
                'salud': 'GET /salud - Estado del servicio',
                'estadisticas': 'GET /estadisticas - Estadísticas'
            },
            'ejemplo': {
                'url': '/predecir',
                'metodo': 'POST',
                'cuerpo': {
                    'titular': 'Noticias importantes desde Silicon Valley',
                    'modelo': 'ensemble'
                }
            }
        })

    @app.route('/predecir', methods=['POST'])
    def predecir():
        """
        Endpoint principal para clasificar titulares.
        USA ENSEMBLE DINÁMICO por defecto si no se especifica modelo.
        """
        if not clasificador_api:
            return jsonify({'error': 'API no inicializada correctamente'}), 500

        try:
            # Obtener datos del request
            data = request.get_json()

            # Validar request
            es_valido, mensaje_error = validate_request(data)
            if not es_valido:
                return jsonify({'error': mensaje_error}), 400

            titular = data['titular'].strip()

            # CAMBIO PRINCIPAL: Si no se especifica modelo, usar ENSEMBLE DINÁMICO
            if 'modelo' not in data:
                nombre_modelo = 'ensemble'  # 🎯 Usa ensemble dinámico por defecto
                logger.info(f"Sin modelo especificado, usando ensemble dinámico para: {titular[:50]}...")
            else:
                nombre_modelo = data['modelo']
                logger.info(f"Modelo especificado: {nombre_modelo} para: {titular[:50]}...")

                # Validar que el modelo existe
                if nombre_modelo not in clasificador_api.models:
                    modelos_disponibles = list(clasificador_api.models.keys())
                    return jsonify({
                        'error': f"Modelo '{nombre_modelo}' no disponible. Disponibles: {modelos_disponibles}"
                    }), 400

            # Realizar predicción
            resultado = clasificador_api.predecir(titular, nombre_modelo)

            logger.info(
                f"Predicción exitosa: {titular[:50]}... -> {resultado['prediccion']['categoria']} (modelo: {resultado['prediccion']['modelo_usado']})")

            return jsonify(resultado)

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error en /predecir: {e}")
            return jsonify({'error': 'Error interno en predicción'}), 500

    @app.route('/modelos', methods=['GET'])
    def modelos():
        """modelos disponibles."""
        if not clasificador_api:
            return jsonify({'error': 'API no inicializada'}), 500

        try:
            stats = get_model_stats()
            modelos_disponibles = {}
            for nombre_modelo in clasificador_api.models.keys():
                if nombre_modelo in stats['modelos_disponibles']:
                    modelos_disponibles[nombre_modelo] = stats['modelos_disponibles'][nombre_modelo]

            mejor_modelo = get_best_model()

            return jsonify({
                'modelos_disponibles': modelos_disponibles,
                'total_modelos': len(clasificador_api.models),
                'recomendado': mejor_modelo,
                'mas_rapido': stats['modelo_mas_rapido'],
                'mejor_precision': stats['mejor_precision'],
                'ollama_disponible': clasificador_api.ollama_available,
                'total_categorias': len(clasificador_api.categories)
            })

        except Exception as e:
            logger.error(f"Error en /modelos: {e}")
            return jsonify({'error': 'Error obteniendo información de modelos'}), 500

    @app.route('/salud', methods=['GET'])
    def salud():
        """Endpoint de health check."""
        if not clasificador_api:
            return jsonify({
                'estado': 'no_saludable',
                'mensaje': 'API no inicializada'
            }), 500

        try:
            titular_prueba = "Titular de prueba para verificar salud"
            resultado_prueba = clasificador_api.predecir(titular_prueba, 'tfidf')

            return jsonify({
                'estado': 'saludable',
                'mensaje': 'API funcionando correctamente',
                'modelos_cargados': len(clasificador_api.models),
                'ollama_disponible': clasificador_api.ollama_available,
                'prediccion_prueba': {
                    'categoria': resultado_prueba['prediccion']['categoria'],
                    'tiempo_procesamiento_ms': resultado_prueba['tiempo_procesamiento_ms']
                }
            })

        except Exception as e:
            logger.error(f"Error en verificación de salud: {e}")
            return jsonify({
                'estado': 'no_saludable',
                'mensaje': f'Error en verificación de salud: {str(e)}'
            }), 500

    @app.route('/estadisticas', methods=['GET'])
    def estadisticas():
        """Obtiene estadísticas generales del servicio."""
        stats = get_model_stats()

        return jsonify({
            'version_api': '1.0.0',
            'total_categorias': len(clasificador_api.categories) if clasificador_api else 0,
            'categorias': clasificador_api.categories if clasificador_api else [],
            'modelos_disponibles': list(clasificador_api.models.keys()) if clasificador_api else [],
            'modelo_recomendado': stats.get('modelo_recomendado', 'tfidf'),
            'mejor_precision': stats.get('mejor_precision', 0),
            'caracteristicas': {
                'clasificacion_tfidf': True,
                'embeddings_semanticos': True,
                'clasificacion_llm': clasificador_api.ollama_available if clasificador_api else False,
                'prediccion_ensemble': True,
                'metricas_dinamicas': True
            }
        })

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )