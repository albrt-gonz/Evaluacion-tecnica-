import json
import pandas as pd
from pathlib import Path


def test_data_loading():
    """
    Script de debug para verificar la carga del dataset
    """
    print("DEBUGGER - CARGA DE DATASET")
    print("=" * 40)

    possible_paths = [
        Path("data/data.json"),
        Path("../data/data.json"),
        Path("config/data/data.json"),
        Path("../config/data/data.json"),
        Path("/Users/imartinezt/PycharmProjects/miprimerflask/config/data/data.json")
    ]

    print("📁 Buscando archivo data.json en:")
    for path in possible_paths:
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {path}")

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if not data_path:
        print("\n❌ No se encontró data.json")
        return False

    print(f"\n✅ Archivo encontrado: {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f" Archivo cargado exitosamente")
        print(f" Tipo de datos: {type(data)}")

        if isinstance(data, dict):
            print(f"  Claves principales: {list(data.keys())[:5]}")

            if 'headline' in data:
                headlines = data['headline']
                print(f"📝 Tipo de headlines: {type(headlines)}")

                if isinstance(headlines, dict):
                    print(f" Número de headlines: {len(headlines)}")
                    print(f" Primeras 3 claves: {list(headlines.keys())[:3]}")
                    print(f" Primer headline: {list(headlines.values())[0][:100]}...")

                    if 'category' in data:
                        categories = data['category']
                        print(f"  Número de categorías: {len(categories)}")
                        print(f" Primeras 3 categorías: {list(categories.values())[:3]}")

                    records = []
                    for idx in headlines.keys():
                        records.append({
                            'headline': headlines[idx],
                            'category': data.get('category', {}).get(idx, 'Unknown')
                        })

                    df = pd.DataFrame(records)
                    print(f"\n DataFrame creado:")
                    print(f"    Dimensiones: {df.shape}")
                    print(f"   ️Categorías únicas: {df['category'].nunique()}")
                    print(f"    Top 3 categorías:")

                    top_cats = df['category'].value_counts().head(3)
                    for cat, count in top_cats.items():
                        print(f"      {cat}: {count}")

                    print(f"\n ¡Dataset procesado correctamente!")
                    return True
                return None
            return None

        elif isinstance(data, list):
            print(f" Lista con {len(data)} elementos")
            if data:
                print(
                    f" Claves del primer elemento: {list(data[0].keys()) if isinstance(data[0], dict) else 'No es dict'}")

            df = pd.DataFrame(data)
            print(f" DataFrame creado: {df.shape}")
            return True
        return None

    except Exception as e:
        print(f" Error procesando archivo: {e}")
        return False


if __name__ == "__main__":
    test_data_loading()