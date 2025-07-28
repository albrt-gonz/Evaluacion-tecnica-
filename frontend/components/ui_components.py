import gradio as gr
from pathlib import Path
import sys

current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from utils.asset_loader import asset_loader


class LiverpoolUIComponents:
    @staticmethod
    def create_input_section():
        """Crea la sección de input principal"""
        with gr.Group(elem_classes=["section-card"]):
            titular_input = gr.Textbox(
                label="📝 Titular de Noticia",
                placeholder="Ej: Tesla stock soars 15% after record quarterly earnings beat expectations",
                lines=3,
                max_lines=5,
                info="Ingresa el titular que deseas clasificar automáticamente",
                elem_classes=["liverpool-input"]
            )

            clasificar_btn = gr.Button(
                "🚀 Clasificar Noticia",
                variant="primary",
                size="lg",
                elem_classes=["liverpool-btn-primary"]
            )

        return titular_input, clasificar_btn

    @staticmethod
    def create_examples_section():
        """Crea la sección de ejemplos con botones"""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML("<h3 class='section-title'>💡 Ejemplos Profesionales</h3>")
            gr.HTML("<p class='section-subtitle'>Haz clic en cualquier ejemplo para probarlo:</p>")

            with gr.Row():
                ejemplo_1 = gr.Button("📈 Tesla earnings record", size="sm", elem_classes=["liverpool-btn-secondary"])
                ejemplo_2 = gr.Button("🌍 California earthquake", size="sm", elem_classes=["liverpool-btn-secondary"])
                ejemplo_3 = gr.Button("🔬 Gene therapy breakthrough", size="sm", elem_classes=["liverpool-btn-secondary"])

            with gr.Row():
                ejemplo_4 = gr.Button("📱 iPhone AI features", size="sm", elem_classes=["liverpool-btn-secondary"])
                ejemplo_5 = gr.Button("⚽ World Cup Final", size="sm", elem_classes=["liverpool-btn-secondary"])
                ejemplo_6 = gr.Button("👶 Parenting sleep tips", size="sm", elem_classes=["liverpool-btn-secondary"])

        return [ejemplo_1, ejemplo_2, ejemplo_3, ejemplo_4, ejemplo_5, ejemplo_6]

    @staticmethod
    def create_status_section():
        """Crea la sección de estado del sistema"""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML("<h4 class='section-title'>📡 Estado del Sistema</h4>")
            estado_api = gr.HTML(
                value=asset_loader.create_loading_state("Verificando conexión...")
            )

        return estado_api

    @staticmethod
    def create_metrics_section():
        """Crea la sección de métricas"""
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML("<h4 class='section-title'>📊 Métricas en Tiempo Real</h4>")
            with gr.Row():
                confianza_display = gr.Textbox(
                    label="Confianza",
                    value="0%",
                    interactive=False,
                    container=False,
                    elem_classes=["liverpool-input"]
                )
                tiempo_display = gr.Textbox(
                    label="Tiempo",
                    value="0 ms",
                    interactive=False,
                    container=False,
                    elem_classes=["liverpool-input"]
                )

        return confianza_display, tiempo_display

    @staticmethod
    def create_results_section():
        """Crea la sección de resultados"""
        gr.HTML(asset_loader.get_section_title("🎯 Resultado de Clasificación"))

        with gr.Row():
            with gr.Column(scale=2):
                resultado_principal = gr.HTML(
                    value=asset_loader.create_empty_state("🔮 Esperando clasificación...")
                )

            with gr.Column(scale=1):
                top_3_display = gr.HTML(
                    value=asset_loader.create_empty_state("⏳ Pendiente...")
                )

        return resultado_principal, top_3_display

    @staticmethod
    def create_dataset_stats_section():
        """Crea la sección de estadísticas del dataset"""
        dataset_stats = gr.HTML(
            value=asset_loader.create_loading_state("Cargando estadísticas...")
        )
        return dataset_stats

    @staticmethod
    def create_dataset_categories_section():
        """Crea la sección de categorías del dataset"""
        gr.HTML(asset_loader.get_section_title("🏆 Categorías Principales"))
        top_categories = gr.HTML(
            value=asset_loader.create_loading_state("Cargando categorías...")
        )
        return top_categories

    @staticmethod
    def create_dataset_table_section():
        """Crea la sección de tabla de datos"""
        gr.HTML(asset_loader.get_section_title("📋 Muestra de Datos"))
        sample_table = gr.HTML(
            value=asset_loader.create_loading_state("Cargando muestra...")
        )
        return sample_table

    @staticmethod
    def create_tabs():
        """Crea los tabs principales con estilos Liverpool"""
        with gr.Tabs(elem_classes=["liverpool-tabs"]):
            with gr.TabItem(" Clasificador", elem_classes=["liverpool-tab"]):
                classifier_components = LiverpoolUIComponents._create_classifier_tab()

            with gr.TabItem(" Dataset", elem_classes=["liverpool-tab"]):
                dataset_components = LiverpoolUIComponents._create_dataset_tab()

        return classifier_components, dataset_components

    @staticmethod
    def _create_classifier_tab():
        """Crea el contenido del tab clasificador"""
        with gr.Row():
            with gr.Column(scale=2):
                titular_input, clasificar_btn = LiverpoolUIComponents.create_input_section()
                ejemplos = LiverpoolUIComponents.create_examples_section()

            with gr.Column(scale=1):
                estado_api = LiverpoolUIComponents.create_status_section()
                confianza_display, tiempo_display = LiverpoolUIComponents.create_metrics_section()

        resultado_principal, top_3_display = LiverpoolUIComponents.create_results_section()

        return {
            'titular_input': titular_input,
            'clasificar_btn': clasificar_btn,
            'ejemplos': ejemplos,
            'estado_api': estado_api,
            'confianza_display': confianza_display,
            'tiempo_display': tiempo_display,
            'resultado_principal': resultado_principal,
            'top_3_display': top_3_display
        }

    @staticmethod
    def _create_dataset_tab():
        """Crea el contenido del tab dataset"""
        gr.HTML(asset_loader.get_dataset_header())

        dataset_stats = LiverpoolUIComponents.create_dataset_stats_section()
        top_categories = LiverpoolUIComponents.create_dataset_categories_section()
        sample_table = LiverpoolUIComponents.create_dataset_table_section()

        return {
            'dataset_stats': dataset_stats,
            'top_categories': top_categories,
            'sample_table': sample_table
        }

    @staticmethod
    def setup_example_events(ejemplos, titular_input, examples_list):
        """Configura los eventos de los botones de ejemplo"""
        for i, btn in enumerate(ejemplos):
            if i < len(examples_list):
                btn.click(
                    lambda example=examples_list[i]: example,
                    outputs=titular_input
                )