#!/usr/bin/env python3
"""
Script principal para ejecutar la API del clasificador de noticias.

REQUISITOS PREVIOS:
1. Archivo data/data.json debe existir
2. Modelos deben estar entrenados: python train.py
3. Ollama instalado (opcional): ollama pull gemma:2b

Uso:
    python run.py

La API estará disponible en: http://localhost:5000
"""

import logging
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def check_requirements():
    print("Verificando requisitos...")

    errors = []
    warnings = []
    data_file = current_dir / "data" / "data.json"
    if not data_file.exists():
        errors.append(f" Archivo de datos no encontrado: {data_file}")
        errors.append("    Coloca tu archivo data.json en la carpeta data/")
    else:
        print(f"Archivo de datos encontrado: {data_file}")

    model_files = [
        "models/tfidf_vectorizer.pkl",
        "models/tfidf_model.pkl",
        "models/sentence_transformer.pkl"
    ]

    missing_models = []
    for model_file in model_files:
        model_path = current_dir / model_file
        if not model_path.exists():
            missing_models.append(model_file)
        else:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f" Modelo encontrado: {model_file} ({size_mb:.1f} MB)")

    if missing_models:
        errors.append("❌ Modelos no entrenados:")
        for model in missing_models:
            errors.append(f"   - {model}")
        errors.append("   Ejecuta: python train.py")

    try:
        import flask, pandas, sklearn, sentence_transformers
        print(" Dependencias Python instaladas")
    except ImportError as e:
        errors.append(f" Dependencia faltante: {e}")
        errors.append("   Ejecuta: pip install -r requirements.txt")

    try:
        import ollama
        ollama.list()
        print(" Ollama disponible (Gemma LLM habilitado)")
    except Exception:
        warnings.append("  Ollama no disponible (Gemma LLM deshabilitado)")
        warnings.append("    Para habilitar: ollama pull gemma:2b")

    if warnings:
        print("\n Advertencias:")
        for warning in warnings:
            print(warning)

    if errors:
        print("\n Errores encontrados:")
        for error in errors:
            print(error)
        return False

    print("\n Todos los requisitos verificados correctamente")
    return True


def main():
    """Función principal."""
    print(" NEWS CLASSIFIER API")
    print("=" * 50)

    # Configurar logging
    setup_logging()

    # Verificar requisitos
    if not check_requirements():
        print("\n No se puede iniciar la API. Corrige los errores arriba.")
        return False

    try:
        # Importar y crear la aplicación
        from app import create_app
        from config.config import API_CONFIG

        app = create_app()

        # Información del servidor
        host = API_CONFIG['host']
        port = API_CONFIG['port']
        debug = API_CONFIG['debug']

        print(f"\n{'=' * 50}")
        print(" INFORMACIÓN DEL SERVIDOR")
        print(f"{'=' * 50}")
        print(f" URL: http://{host}:{port}")
        print(f" Docs: http://localhost:{port}/")
        print(f" Debug: {debug}")

        print(f"\n{'=' * 50}")
        print(" ENDPOINTS DISPONIBLES")
        print(f"{'=' * 50}")
        print("POST /predict        - Clasificar titular")
        print("GET  /models         - Modelos disponibles")
        print("GET  /health         - Estado del servicio")
        print("GET  /stats          - Estadísticas")
        print("GET  /               - Documentación")

        print(f"\n{'=' * 50}")
        print(" EJEMPLO DE USO")
        print(f"{'=' * 50}")
        print(f"""curl -X POST http://localhost:{port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"headline": "Breaking news from California", "model": "ensemble"}}'""")

        print(f"\n{'=' * 50}")
        print(" ¡Servidor iniciando! Presiona Ctrl+C para detener")
        print(f"{'=' * 50}")

        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )

    except KeyboardInterrupt:
        print("\n\n ¡Servidor detenido por el usuario!")
        return True

    except ImportError as e:
        print(f"\n Error importando módulos: {e}")
        print("💡 Asegúrate de que todos los archivos estén en su lugar")
        return False

    except Exception as e:
        print(f"\n Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)