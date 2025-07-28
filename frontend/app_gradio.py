import gradio as gr
import sys
import json
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.api_client import api_client
from utils.category_mapper import category_mapper


def create_top3_card(top_3_data, full_result):
    """Card para top 3 categorías con información de modelos alternativos"""
    if not top_3_data:
        return "<div style='text-align: center; color: #6b7280; padding: 2rem;'>No hay datos disponibles</div>"

    # Top 3 categorías
    items_html = ""
    for i, (cat, prob) in enumerate(list(top_3_data.items())[:3], 1):
        items_html += f"""
        <div style="display: flex; align-items: center; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; background: #f9fafb;">
            <div style="width: 24px; height: 24px; background: #ec008c; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">{i}</div>
            <div style="flex: 1; font-weight: 500; color: #101820;">{cat}</div>
            <div style="font-weight: 600; color: #ec008c;">{prob:.1%}</div>
        </div>
        """

    # Información de modelos alternativos
    alternatives_html = ""
    if 'modelos_alternativos' in full_result:
        alternatives = full_result['modelos_alternativos']
        alternatives_html = """
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(83, 86, 90, 0.05); border-radius: 8px;">
            <h5 style="margin: 0 0 0.5rem 0; color: #53565a; font-size: 0.9rem;">🤖 Otros Modelos</h5>
        """

        for model_key, alt_data in alternatives.items():
            model_display = model_key.replace('_', '-').upper()
            categoria_alt = alt_data.get('categoria', 'N/A')
            confianza_alt = alt_data.get('confianza', 0)
            tiempo_alt = alt_data.get('tiempo_ms', 0)

            alternatives_html += f"""
            <div style="font-size: 0.8rem; color: #53565a; margin-bottom: 0.25rem;">
                <strong>{model_display}:</strong> {categoria_alt} ({confianza_alt:.1%}) - {tiempo_alt:.1f}ms
            </div>
            """

        alternatives_html += "</div>"

    return f"""
    <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
        <h4 style="font-size: 1.1rem; font-weight: 600; color: #101820; margin: 0 0 1rem 0;">🏆 Top 3 Categorías</h4>
        <div>{items_html}</div>
        {alternatives_html}
    </div>
    """


def create_error_card(message):
    """Crea card de error"""
    return f"""
    <div style="background: rgba(239, 68, 68, 0.05); border: 2px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
        <div style="color: #ef4444; font-weight: 500;">{message}</div>
    </div>
    """


def get_confidence_color(confidence_value):
    """Retorna color basado en nivel de confianza"""
    if confidence_value >= 0.8:
        return "#10b981"
    elif confidence_value >= 0.6:
        return "#f59e0b"
    elif confidence_value >= 0.4:
        return "#ec008c"
    else:
        return "#ef4444"


def get_professional_examples() -> list:
    """Ejemplos profesionales de titulares"""
    return [
        "Tesla stock soars 15% after record quarterly earnings beat expectations",
        "Breaking: Major 7.2 earthquake hits California coast, tsunami warning issued",
        "Scientists discover revolutionary gene therapy treatment for Type 1 diabetes",
        "Apple unveils iPhone 16 with groundbreaking AI features and 48-hour battery life",
        "World Cup Final: Argentina defeats France 4-2 in most-watched match in history",
        "Parents share genius bedtime routine that gets kids to sleep in under 10 minutes"
    ]


class NewsClassifierGradioApp:
    """
    Interfaz Gradio para el Clasificador de Noticias
    """

    def __init__(self):
        self.api_client = api_client
        self.category_mapper = category_mapper
        self.app = None

    def predict_news_category(self, titular: str) -> tuple:
        """Función principal de predicción con métricas dinámicas"""
        if not titular or not titular.strip():
            return (
                create_error_card("Por favor, ingresa un titular válido"),
                "0%",
                "0 ms",
                "Esperando clasificación...",
                "❌ Entrada vacía"
            )

        is_healthy, health_message = self.api_client.check_health()
        if not is_healthy:
            return (
                create_error_card(f"Error de conexión: {health_message}"),
                "0%",
                "0 ms",
                "API no disponible",
                f"❌ {health_message}"
            )
        result = self.api_client.predict(titular)

        if result.get('error', False):
            error_msg = result.get('error_message', 'Error desconocido')
            return (
                create_error_card(f"Error: {error_msg}"),
                "0%",
                "0 ms",
                "Error en predicción",
                f"❌ {error_msg}"
            )

        formatted_result = self.category_mapper.format_result(result)
        categoria = formatted_result['prediccion']['categoria_formateada']
        confianza = formatted_result['prediccion']['confianza']
        tiempo_total = formatted_result.get('tiempo_total_ms', 0)
        modelo_usado = formatted_result['prediccion']['modelo_usado']

        resultado_html = self.create_result_card(categoria, confianza, modelo_usado, tiempo_total, formatted_result)
        top_3_html = create_top3_card(formatted_result.get('top_3_formateadas', {}), formatted_result)

        return (
            resultado_html,
            f"{confianza:.1%}",
            f"{tiempo_total:.1f} ms",
            top_3_html,
            " API funcionando correctamente"
        )

    def create_result_card(self, categoria, confianza, modelo, tiempo, full_result):
        """Crea card de resultado con información de evaluación dinámica"""
        confidence_color = get_confidence_color(confianza)
        evaluation_html = ""
        if 'evaluacion_dinamica' in full_result:
            eval_info = full_result['evaluacion_dinamica']
            modelos_evaluados = eval_info.get('modelos_evaluados', 1)
            resumen = eval_info.get('resumen_confianzas', {})
            ganador = eval_info.get('ganador', {})

            evaluation_html = f"""
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(236, 0, 140, 0.05); border-radius: 8px; border-left: 4px solid #ec008c;">
                <h5 style="margin: 0 0 0.5rem 0; color: #ec008c; font-size: 0.9rem;">🎯 Evaluación Dinámica</h5>
                <p style="margin: 0; font-size: 0.8rem; color: #53565a;">
                    <strong>Modelos evaluados:</strong> {modelos_evaluados}/3 disponibles<br>
                    <strong>🏆 Ganador:</strong> {ganador.get('modelo', 'N/A').upper()} ({ganador.get('confianza', 0):.1%})<br>
            """

            for model_key, confidence_val in resumen.items():
                if confidence_val is not None:
                    model_name = model_key.replace('_', '-').upper()
                    winner_mark = "🥇" if model_key == ganador.get('modelo', '').lower() else "📊"
                    evaluation_html += f"<strong>{winner_mark} {model_name}:</strong> {confidence_val:.1%} | "

            evaluation_html = evaluation_html.rstrip(" | ") + "</p></div>"

        return f"""
        <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.1); border: 1px solid #d1d3d4; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #d1d3d4;">
                <h3 style="font-size: 1.5rem; font-weight: 600; color: #101820; margin: 0;">{categoria}</h3>
                <div style="padding: 0.5rem 1rem; border-radius: 50px; color: white; font-weight: 600; background-color: {confidence_color};">
                    {confianza:.1%}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="text-align: center; padding: 1rem; background: #f9fafb; border-radius: 8px;">
                    <span style="display: block; font-size: 0.8rem; color: #53565a; font-weight: 500; text-transform: uppercase; margin-bottom: 0.25rem;">Modelo Ganador</span>
                    <span style="display: block; font-size: 1.1rem; color: #101820; font-weight: 600;">{modelo.upper()}</span>
                </div>
                <div style="text-align: center; padding: 1rem; background: #f9fafb; border-radius: 8px;">
                    <span style="display: block; font-size: 0.8rem; color: #53565a; font-weight: 500; text-transform: uppercase; margin-bottom: 0.25rem;">Tiempo</span>
                    <span style="display: block; font-size: 1.1rem; color: #101820; font-weight: 600;">{tiempo:.1f} ms</span>
                </div>
            </div>
            {evaluation_html}
        </div>
        """

    def load_data_preview(self):
        """Carga vista previa del dataset"""
        try:
            possible_paths = [
                Path("data/data.json"),
                Path("../data/data.json"),
                Path("config/data/data.json"),
                Path("../config/data/data.json"),
                Path("/Users/imartinezt/PycharmProjects/miprimerflask/config/data/data.json")
            ]

            data_path = None
            for path in possible_paths:
                if path.exists():
                    data_path = path
                    break

            if not data_path:
                return "❌ Dataset no encontrado", "", ""

            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'headline' in data and 'category' in data:
                headlines_dict = data['headline']
                categories_dict = data['category']

                indices = sorted(headlines_dict.keys(), key=lambda x: int(x))
                headlines = [headlines_dict[idx] for idx in indices]
                categories = [categories_dict[idx] for idx in indices]

                df = pd.DataFrame({
                    'headline': headlines,
                    'category': categories
                })
            else:
                df = pd.DataFrame(data)

            total_noticias = len(df)
            total_categorias = df['category'].nunique()

            stats_html = f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 2rem; font-weight: 700; color: #ec008c;">{total_noticias:,}</div>
                    <div style="font-size: 0.9rem; color: #53565a;">Total Noticias</div>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 2rem; font-weight: 700; color: #ec008c;">{total_categorias}</div>
                    <div style="font-size: 0.9rem; color: #53565a;">Categorías</div>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 2rem; font-weight: 700; color: #ec008c;">2012-2022</div>
                    <div style="font-size: 0.9rem; color: #53565a;">Período</div>
                </div>
            </div>
            """

            top_categories = df['category'].value_counts().head(8)
            categories_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;'>"
            for cat, count in top_categories.items():
                emoji = self.category_mapper.get_emoji(cat)
                percentage = (count / total_noticias) * 100
                categories_html += f"""
                <div style="display: flex; align-items: center; background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 1.5rem; margin-right: 0.75rem;">{emoji}</div>
                    <div>
                        <div style="font-weight: 600; color: #101820;">{cat}</div>
                        <div style="font-size: 0.9rem; color: #53565a;">{count:,} ({percentage:.1f}%)</div>
                    </div>
                </div>
                """
            categories_html += "</div>"
            sample_data = df.sample(n=min(10, len(df)), random_state=42)
            table_html = """
            <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                <div style="background: #ec008c; display: grid; grid-template-columns: 60px 2fr 1fr; padding: 1rem; color: white; font-weight: 600;">
                    <div>ID</div><div>Titular</div><div>Categoría</div>
                </div>
            """

            for idx, (_, row) in enumerate(sample_data.iterrows(), 1):
                emoji = self.category_mapper.get_emoji(row['category'])
                headline = (row['headline'][:80] + '...') if len(row['headline']) > 80 else row['headline']
                table_html += f"""
                <div style="display: grid; grid-template-columns: 60px 2fr 1fr; padding: 1rem; border-bottom: 1px solid #f3f4f6;">
                    <div style="text-align: center; color: #ec008c; font-weight: 600;">{idx}</div>
                    <div style="color: #101820;">{headline}</div>
                    <div style="color: #ec008c;">{emoji} {row['category']}</div>
                </div>
                """

            table_html += "</div>"

            return stats_html, categories_html, table_html

        except Exception as e:
            return f"❌ Error: {e}", "", ""

    def create_interface(self):
        """Crea la interfaz Gradio"""

        css = """
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
        }
        .liverpool-header {
            background: linear-gradient(135deg, #ec008c 0%, #53565a 100%);
            padding: 3rem 2rem;
            border-radius: 16px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(236, 0, 140, 0.15);
        }
        .section-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #d1d3d4;
            margin-bottom: 1rem;
        }
        """

        with gr.Blocks(
            title="📰 Liverpool News Classifier",
            theme=gr.themes.Soft(),
            css=css
        ) as self.app:

            # Header
            gr.HTML("""
            <div class="liverpool-header">
                <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">Clasificador de Noticias</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">Evaluación dinámica de múltiples modelos de Machine Learning</p>
                <p style="font-size: 1rem; opacity: 0.8;">Alberto Gonzalez | TEST | 27 Julio</p>
            </div>
            """)

            with gr.Tabs():
                with gr.TabItem("🎯 Clasificador"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["section-card"]):
                                titular_input = gr.Textbox(
                                    label="📝 Titular de Noticia",
                                    placeholder="Ej: Tesla stock soars 15% after record quarterly earnings",
                                    lines=3,
                                    info="El sistema evaluará automáticamente los 3 modelos disponibles"
                                )
                                clasificar_btn = gr.Button("🚀 Clasificar Noticia", variant="primary", size="lg")

                            with gr.Group(elem_classes=["section-card"]):
                                gr.HTML("<h3>Ejemplos: </h3>")
                                with gr.Row():
                                    ej1 = gr.Button("📈 Tesla earnings", size="sm")
                                    ej2 = gr.Button("🌍 Earthquake news", size="sm")
                                    ej3 = gr.Button("🔬 Gene therapy", size="sm")
                                with gr.Row():
                                    ej4 = gr.Button("📱 iPhone AI", size="sm")
                                    ej5 = gr.Button("⚽ World Cup", size="sm")
                                    ej6 = gr.Button("👶 Parenting tips", size="sm")

                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["section-card"]):
                                gr.HTML("<h4>📊 Métricas y Monitoreo </h4>")
                                confianza_display = gr.Textbox(label="Confianza", value="0%", interactive=False)
                                tiempo_display = gr.Textbox(label="Tiempo", value="0 ms", interactive=False)
                                estado_api = gr.Textbox(label="Monitoreo", value="Esperando...", interactive=False)

                    gr.HTML("<h2 style='text-align: center; color: #101820; margin: 2rem 0;'>Resultados de la Clasificación</h2>")

                    with gr.Row():
                        resultado_principal = gr.HTML(value="<div style='text-align: center; padding: 2rem; color: #6b7280;'>🔮 Esperando clasificación...</div>")
                        top_3_display = gr.HTML(value="<div style='text-align: center; padding: 2rem; color: #6b7280;'>⏳ Pendiente...</div>")

                with gr.TabItem("📊 Dataset"):
                    gr.HTML("""
                    <div class="liverpool-header">
                        <h2>📊 Explorador del Dataset</h2>
                        <p>Visualiza y analiza los datos de entrenamiento</p>
                    </div>
                    """)

                    dataset_stats = gr.HTML(value="🔄 Cargando...")
                    gr.HTML("<h3 style='color: #101820; margin: 2rem 0 1rem 0;'>🏆 Top Categorías</h3>")
                    top_categories = gr.HTML(value="🔄 Cargando...")
                    gr.HTML("<h3 style='color: #101820; margin: 2rem 0 1rem 0;'>📋 Muestra de Datos</h3>")
                    sample_table = gr.HTML(value="🔄 Cargando...")

            # Footer
            gr.HTML("""
            <div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #f9fafb, #f3f4f6); border-radius: 16px; text-align: center;">
                <h3 style="color: #101820;">🎯 Sistema con Evaluación Dinámica</h3>
                <p style="color: #53565a;"><strong>Modelos:</strong> TF-IDF • Sentence-Transformers • Gemma LLM</p>
                <p style="color: #8a8d91; font-size: 0.9rem;">⚡ Evalúa automáticamente • 🏆 Selecciona el mejor • 📊 Métricas transparentes</p>
            </div>
            """)

            # Eventos
            clasificar_btn.click(
                fn=self.predict_news_category,
                inputs=[titular_input],
                outputs=[resultado_principal, confianza_display, tiempo_display, top_3_display, estado_api]
            )

            # Ejemplos
            ejemplos = get_professional_examples()
            ej1.click(lambda: ejemplos[0], outputs=titular_input)
            ej2.click(lambda: ejemplos[1], outputs=titular_input)
            ej3.click(lambda: ejemplos[2], outputs=titular_input)
            ej4.click(lambda: ejemplos[3], outputs=titular_input)
            ej5.click(lambda: ejemplos[4], outputs=titular_input)
            ej6.click(lambda: ejemplos[5], outputs=titular_input)

            # Cargar datos
            self.app.load(
                fn=self.load_data_preview,
                outputs=[dataset_stats, top_categories, sample_table]
            )

        return self.app

    def launch(self, **kwargs):
        """Lanza la aplicación"""
        if not self.app:
            self.create_interface()

        default_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'show_error': True
        }
        default_kwargs.update(kwargs)

        print(" Iniciando Clasificador Liverpool México...")
        print(f" URL: http://localhost:{default_kwargs['server_port']}")

        return self.app.launch(**default_kwargs)


def main():
    """Función principal"""
    print("📰 LIVERPOOL MÉXICO - NEWS CLASSIFIER")
    print("=" * 45)
    print("🎨 Paleta: Rosa Liverpool #ec008c")
    print("🎯 Evaluación: Dinámica de 3 modelos")
    print("=" * 45)

    app = NewsClassifierGradioApp()

    try:
        app.launch()
        return None
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta pronto!")
        return True


if __name__ == "__main__":
    main()