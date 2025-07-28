"""
Estos tests verifican que los endpoints principales funcionan correctamente.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app


@pytest.fixture
def app():
    """Fixture para crear la app de test."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Fixture para el cliente de test."""
    return app.test_client()


def test_home_endpoint(client):
    """Test del endpoint principal."""
    response = client.get('/')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'message' in data
    assert 'endpoints' in data
    assert 'News Headline Classifier' in data['message']


def test_health_check(client):
    """Test del health check."""
    response = client.get('/health')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'status' in data
    assert 'models' in data
    assert data['status'] in ['healthy', 'degraded']


def test_models_info(client):
    """Test del endpoint de información de modelos."""
    response = client.get('/models')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'available_models' in data
    assert len(data['available_models']) > 0


def test_stats_endpoint(client):
    """Test del endpoint de estadísticas."""
    response = client.get('/stats')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'models_loaded' in data
    assert 'total_models' in data


def test_predict_endpoint_valid(client):
    """Test de predicción con datos válidos."""
    test_data = {
        'headline': 'Breaking news from technology sector',
        'model': 'tfidf'  # Usar TF-IDF si está disponible
    }

    response = client.post('/predict',
                           data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code in [200, 400]

    data = json.loads(response.data)

    if response.status_code == 200:
        assert 'headline' in data
        assert 'prediction' in data
        assert 'category' in data['prediction']
        assert 'confidence' in data['prediction']
    else:
        assert 'error' in data


def test_predict_endpoint_invalid_json(client):
    """Test con JSON inválido."""
    response = client.post('/predict',
                           data='invalid json',
                           content_type='application/json')

    assert response.status_code == 400


def test_predict_endpoint_missing_headline(client):
    """Test sin headline."""
    test_data = {'model': 'tfidf'}

    response = client.post('/predict',
                           data=json.dumps(test_data),
                           content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_endpoint_empty_headline(client):
    """Test con headline vacío."""
    test_data = {'headline': '', 'model': 'tfidf'}

    response = client.post('/predict',
                           data=json.dumps(test_data),
                           content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_endpoint_invalid_model(client):
    """Test con modelo inválido."""
    test_data = {
        'headline': 'Test headline',
        'model': 'invalid_model'
    }

    response = client.post('/predict',
                           data=json.dumps(test_data),
                           content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'not available' in data['error']


def test_predict_batch_endpoint(client):
    """Test del endpoint de predicción en batch."""
    test_data = {
        'headlines': [
            'Breaking news story',
            'Funny cat video goes viral',
            'New parenting tips revealed'
        ],
        'model': 'tfidf'
    }

    response = client.post('/predict/batch',
                           data=json.dumps(test_data),
                           content_type='application/json')

    assert response.status_code in [200, 400]

    data = json.loads(response.data)

    if response.status_code == 200:
        assert 'results' in data
        assert 'total_processed' in data
        assert len(data['results']) == 3
    else:
        assert 'error' in data


def test_predict_batch_too_many_headlines(client):
    """Test con demasiados headlines en batch."""
    test_data = {
        'headlines': ['test'] * 101,  # Más del límite
        'model': 'tfidf'
    }

    response = client.post('/predict/batch',
                           data=json.dumps(test_data),
                           content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Maximum 100 headlines' in data['error']


def test_404_handler(client):
    """Test del handler de 404."""
    response = client.get('/nonexistent')
    assert response.status_code == 404

    data = json.loads(response.data)
    assert 'error' in data
    assert 'not found' in data['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])