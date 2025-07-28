"""
Cargador de assets (CSS, HTML, etc.) para la interfaz Liverpool México
Versión corregida con evaluación dinámica
"""

from pathlib import Path
import sys

# Agregar paths necesarios
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

try:
    from assets.config.colors import LiverpoolColors
    from assets.templates.templates import LiverpoolTemplates
except ImportError:
    # Fallback si no existen los archivos
    class LiverpoolColors:
        ROSA_LIVERPOOL = "#ec008c"
        SUCCESS = "#10b981"
        WARNING = "#f59e0b"
        ERROR = "#ef4444"

        @classmethod
        def get_confidence_color(cls, confidence_value):
            if confidence_value >= 0.8:
                return cls.SUCCESS
            elif confidence_value >= 0.6:
                return cls.WARNING
            elif confidence_value >= 0.4:
                return cls.ROSA_LIVERPOOL
            else:
                return cls.ERROR

        @classmethod
        def get_css_variables(cls):
            return f"""
            :root {{
                --liverpool-rosa: {cls.ROSA_LIVERPOOL};
                --liverpool-rosa-hover: #d1007a;
                --liverpool-rosa-light: #f5e6f0;
                --liverpool-gris-oscuro: #53565a;
                --liverpool-gris-claro: #d1d3d4;
                --liverpool-negro: #101820;
                --liverpool-blanco: #ffffff;
                --color-success: {cls.SUCCESS};
                --color-warning: {cls.WARNING};
                --color-error: {cls.ERROR};
            }}
            """

    class LiverpoolTemplates:
        @staticmethod
        def create_result_card(categoria, confianza, modelo, tiempo, evaluation_info=None):
            confidence_color = LiverpoolColors.get_confidence_color(confianza)
            evaluation_html = ""
            if evaluation_info:
                modelos_evaluados = evaluation_info.get('modelos_evaluados', 1)
                resumen = evaluation_info.get('resumen_confianzas', {})

                evaluation_html = f"""
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(236, 0, 140, 0.05); border-radius: 8px; border-left: 4px solid #ec008c;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #ec008c; font-size: 0.9rem;">🎯 Evaluación Dinámica</h5>
                    <p style="margin: 0; font-size: 0.8rem; color: #53565a;">
                        <strong>Modelos evaluados:</strong> {modelos_evaluados}/3 disponibles<br>
                """

                model_names = {'tfidf': 'TF-IDF', 'sentence_transformer': 'Sentence-T', 'gemma': 'Gemma LLM'}
                for model_key, confidence_val in resumen.items():
                    if confidence_val is not None:
                        model_display = model_names.get(model_key, model_key)
                        winner_mark = "🏆" if model_key == modelo.lower() else "📊"
                        evaluation_html += f"<strong>{winner_mark} {model_display}:</strong> {confidence_val:.3f} | "

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

        @staticmethod
        def create_top3_card(top_3_data, alternatives_data=None):
            if not top_3_data:
                return "<div style='text-align: center; color: #6b7280; padding: 2rem;'>No hay datos disponibles</div>"

            items_html = ""
            for i, (cat, prob) in enumerate(list(top_3_data.items())[:3], 1):
                items_html += f"""
                <div style="display: flex; align-items: center; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; background: #f9fafb;">
                    <div style="width: 24px; height: 24px; background: #ec008c; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.75rem;">{i}</div>
                    <div style="flex: 1; font-weight: 500; color: #101820;">{cat}</div>
                    <div style="font-weight: 600; color: #ec008c;">{prob:.1%}</div>
                </div>
                """

            alternatives_html = ""
            if alternatives_data:
                alternatives_html = """
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(83, 86, 90, 0.05); border-radius: 8px;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #53565a; font-size: 0.9rem;">🤖 Otros Modelos</h5>
                """

                model_names = {'tfidf': 'TF-IDF', 'sentence_transformer': 'Sentence-T', 'gemma': 'Gemma'}
                for model_key, alt_data in alternatives_data.items():
                    model_display = model_names.get(model_key, model_key)
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

        @staticmethod
        def get_main_header():
            return """
            <div style="background: linear-gradient(135deg, #ec008c 0%, #53565a 100%); padding: 3rem 2rem; border-radius: 16px; text-align: center; color: white; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(236, 0, 140, 0.15);">
                <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">📰 Clasificador Inteligente de Noticias</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">Evaluación dinámica de múltiples modelos de Machine Learning</p>
                <p style="font-size: 1rem; opacity: 0.8;">🎯 Evalúa TF-IDF, Sentence-Transformers y Gemma LLM • Selecciona automáticamente el mejor</p>
            </div>
            """

        @staticmethod
        def get_footer():
            return """
            <div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #f9fafb, #f3f4f6); border-radius: 16px; text-align: center;">
                <h3 style="color: #101820; margin-bottom: 0.5rem;">🎯 Sistema de Clasificación con Evaluación Dinámica</h3>
                <p style="color: #53565a; margin-bottom: 0.5rem;"><strong>Modelos:</strong> TF-IDF • Sentence-Transformers • Gemma LLM • Ensemble Dinámico</p>
                <p style="color: #8a8d91; font-size: 0.9rem;">⚡ Evalúa múltiples modelos en tiempo real • 🏆 Selecciona automáticamente el mejor • 📊 Métricas transparentes</p>
            </div>
            """


def get_templates():
    """Retorna la clase de templates"""
    return LiverpoolTemplates


def get_colors():
    """Retorna la clase de colores"""
    return LiverpoolColors


def create_result_card(categoria, confianza, modelo, tiempo, evaluation_info=None):
    """Crea card de resultado usando templates con evaluación dinámica"""
    return LiverpoolTemplates.create_result_card(categoria, confianza, modelo, tiempo, evaluation_info)


def create_top3_card(top_3_data, alternatives_data=None):
    """Crea card de top 3 usando templates con modelos alternativos"""
    return LiverpoolTemplates.create_top3_card(top_3_data, alternatives_data)


def create_error_card(message, details=None):
    """Crea card de error usando templates"""
    details_html = ""
    if details:
        details_html = f"""
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #53565a;">
            {details}
        </div>
        """

    return f"""
    <div style="background: rgba(239, 68, 68, 0.05); border: 2px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
        <div style="color: #ef4444; font-weight: 500;">{message}</div>
        {details_html}
    </div>
    """


def create_stats_grid(total_noticias, total_categorias, periodo="2012-2022", archivo="data.json"):
    """Crea grid de estadísticas usando templates"""
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
            <div style="font-size: 2rem; font-weight: 700; color: #ec008c; margin-bottom: 0.25rem;">{total_noticias:,}</div>
            <div style="font-size: 0.9rem; color: #53565a; font-weight: 500;">Total Noticias</div>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
            <div style="font-size: 2rem; font-weight: 700; color: #ec008c; margin-bottom: 0.25rem;">{total_categorias}</div>
            <div style="font-size: 0.9rem; color: #53565a; font-weight: 500;">Categorías Únicas</div>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
            <div style="font-size: 2rem; font-weight: 700; color: #ec008c; margin-bottom: 0.25rem;">{periodo}</div>
            <div style="font-size: 0.9rem; color: #53565a; font-weight: 500;">Período</div>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
            <div style="font-size: 1rem; font-weight: 700; color: #ec008c; margin-bottom: 0.25rem;">{archivo}</div>
            <div style="font-size: 0.9rem; color: #53565a; font-weight: 500;">Archivo Fuente</div>
        </div>
    </div>
    """


def create_categories_grid(top_categories_data, category_mapper):
    """Crea grid de categorías usando templates"""
    categories_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 2rem;'>"

    for cat, count, percentage in top_categories_data:
        emoji = category_mapper.get_emoji(cat)
        categories_html += f"""
        <div style="display: flex; align-items: center; background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">{emoji}</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #101820; margin-bottom: 0.25rem;">{cat}</div>
                <div style="font-size: 0.9rem; color: #53565a;">{count:,} noticias ({percentage:.1f}%)</div>
            </div>
        </div>
        """

    categories_html += "</div>"
    return categories_html


def create_data_table(sample_data, category_mapper, total_noticias):
    """Crea tabla de datos usando templates"""
    table_html = """
    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #d1d3d4; margin: 1rem 0;">
        <div style="background: #ec008c; display: grid; grid-template-columns: 60px 2fr 1fr; padding: 1rem; font-weight: 600; color: white; font-size: 0.9rem;">
            <div>ID</div>
            <div>Titular</div>
            <div>Categoría</div>
        </div>
    """

    for idx, (headline, category) in enumerate(sample_data, 1):
        emoji = category_mapper.get_emoji(category)
        headline_truncated = (headline[:100] + '...') if len(headline) > 100 else headline

        table_html += f"""
        <div style="display: grid; grid-template-columns: 60px 2fr 1fr; padding: 1rem; border-bottom: 1px solid #d1d3d4;">
            <div style="text-align: center; font-weight: 600; color: #ec008c;">{idx}</div>
            <div style="color: #101820; line-height: 1.4;">{headline_truncated}</div>
            <div style="color: #ec008c; font-weight: 500;">{emoji} {category}</div>
        </div>
        """

    table_html += f"""
    </div>
    <div style="text-align: center; margin-top: 1rem; color: #53565a; font-size: 0.9rem;">
        📋 Mostrando {len(sample_data)} noticias aleatorias de {total_noticias:,} total
    </div>
    """

    return table_html


def create_status_display(status_type, message):
    """Crea display de estado usando templates"""
    status_styles = {
        'success': 'background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2);',
        'error': 'background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2);',
        'warning': 'background: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.2);',
        'info': 'background: rgba(236, 0, 140, 0.1); color: #ec008c; border: 1px solid rgba(236, 0, 140, 0.2);'
    }

    icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }

    style = status_styles.get(status_type, status_styles['info'])
    icon = icons.get(status_type, 'ℹ️')

    return f"""
    <div style="{style} padding: 0.75rem; border-radius: 8px; font-weight: 500;">
        {icon} {message}
    </div>
    """


def get_confidence_color(confidence_value):
    """Obtiene color basado en nivel de confianza"""
    return LiverpoolColors.get_confidence_color(confidence_value)


def get_main_header():
    """Obtiene header principal"""
    return LiverpoolTemplates.get_main_header()


def get_dataset_header():
    """Obtiene header del dataset"""
    return """
    <div style="background: linear-gradient(135deg, #ec008c 0%, #53565a 100%); padding: 3rem 2rem; border-radius: 16px; text-align: center; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0 0 0.5rem 0; font-size: 2rem;">📊 Explorador del Dataset</h2>
        <p style="margin: 0; opacity: 0.9;">Visualiza y analiza los datos de entrenamiento</p>
    </div>
    """


def get_footer():
    """Obtiene footer"""
    return LiverpoolTemplates.get_footer()


def get_section_title(title, subtitle=None):
    """Obtiene título de sección"""
    subtitle_html = f"<p style='color: #53565a; margin: 0 0 1rem 0;'>{subtitle}</p>" if subtitle else ""
    return f"""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #101820; display: flex; align-items: center; gap: 0.5rem;">{title}</h2>
        {subtitle_html}
    </div>
    """


def create_loading_state(message="Cargando..."):
    """Crea estado de carga"""
    return f"""
    <div style="text-align: center; color: #53565a; padding: 2rem; font-style: italic;">
        🔄 {message}
    </div>
    """


def create_empty_state(message="No hay datos disponibles"):
    """Crea estado vacío"""
    return f"""
    <div style="text-align: center; color: #53565a; padding: 2rem; font-style: italic; background: rgba(209, 211, 212, 0.1); border-radius: 12px; border: 2px dashed #d1d3d4;">
        📭 {message}
    </div>
    """


class AssetLoader:
    """Cargador centralizado de assets para la interfaz"""

    def __init__(self):
        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.css_dir = self.assets_dir / "css"
        self.templates_dir = self.assets_dir / "templates"
        self._ensure_directories()

    def _ensure_directories(self):
        for directory in [self.assets_dir, self.css_dir, self.templates_dir]:
            directory.mkdir(exist_ok=True)

    def load_css(self):
        css_content = ""
        css_content += LiverpoolColors.get_css_variables()
        css_content += "\n\n"

        # CSS principal
        main_css_file = self.css_dir / "main.css"
        if main_css_file.exists():
            with open(main_css_file, 'r', encoding='utf-8') as f:
                css_content += f.read()
                css_content += "\n\n"

        # CSS de componentes
        components_css_file = self.css_dir / "components.css"
        if components_css_file.exists():
            with open(components_css_file, 'r', encoding='utf-8') as f:
                css_content += f.read()

        return css_content

asset_loader = AssetLoader()