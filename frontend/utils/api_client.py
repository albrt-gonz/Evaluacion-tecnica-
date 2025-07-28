import time
from typing import Dict, Tuple

import requests


class NewsClassifierAPIClient:
    """Cliente para la API de clasificación de noticias"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 30

    def check_health(self) -> Tuple[bool, str]:
        """
        Verifica si la API está funcionando

        Returns:
            Tuple[bool, str]: (is_healthy, message)
        """
        try:
            response = requests.get(
                f"{self.base_url}/salud",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return True, data.get('mensaje', 'API funcionando')
            else:
                return False, f"API respondió con código {response.status_code}"

        except requests.ConnectionError:
            return False, "No se puede conectar a la API. ¿Está ejecutándose?"
        except requests.Timeout:
            return False, "Timeout conectando con la API"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"

    def get_available_models(self) -> Dict:
        """
        Obtiene información sobre los modelos disponibles

        Returns:
            Dict: Información de modelos
        """
        try:
            response = requests.get(
                f"{self.base_url}/modelos",
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'Error {response.status_code}',
                    'modelos_disponibles': {},
                    'recomendado': 'auto'
                }

        except Exception as e:
            return {
                'error': str(e),
                'modelos_disponibles': {},
                'recomendado': 'auto'
            }

    def predict(self, titular: str, modelo: str = None) -> Dict:
        """
        Realiza predicción de categoría

        Args:
            titular (str): Titular a clasificar
            modelo (str): Modelo a usar (None = automático)

        Returns:
            Dict: Resultado de la predicción
        """
        try:
            payload = {"titular": titular.strip()}
            if modelo and modelo != "auto":
                payload["modelo"] = modelo

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predecir",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            request_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()

                result['tiempo_total_ms'] = (
                    request_time + result.get('tiempo_procesamiento_ms', 0)
                )

                return result
            else:
                error_data = response.json() if response.content else {}
                return {
                    'error': True,
                    'error_message': error_data.get('error', f'Error HTTP {response.status_code}'),
                    'titular': titular,
                    'status_code': response.status_code
                }

        except requests.ConnectionError:
            return {
                'error': True,
                'error_message': 'No se puede conectar al backend. ¿Está ejecutándose?',
                'titular': titular
            }
        except requests.Timeout:
            return {
                'error': True,
                'error_message': 'Timeout - La predicción tardó demasiado',
                'titular': titular
            }
        except Exception as e:
            return {
                'error': True,
                'error_message': f'Error inesperado: {str(e)}',
                'titular': titular
            }

    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del servicio
        """
        try:
            response = requests.get(
                f"{self.base_url}/estadisticas",
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Error {response.status_code}'}

        except Exception as e:
            return {'error': str(e)}

api_client = NewsClassifierAPIClient()