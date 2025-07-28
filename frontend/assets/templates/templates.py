class LiverpoolTemplates:
    """
    Templates HTML para la interfaz Liverpool México
    Versión actualizada para mostrar evaluación dinámica de modelos
    """

    @staticmethod
    def get_main_header():
        """Header principal de la aplicación"""
        return """
        <div class="main-header">
            <h1 class="main-title">📰 Clasificador Inteligente de Noticias</h1>
            <p class="main-subtitle">Evaluación dinámica de múltiples modelos de Machine Learning</p>
            <p class="main-description">🎯 Evalúa TF-IDF, Sentence-Transformers y Gemma LLM • Selecciona automáticamente el mejor</p>
        </div>
        """

    @staticmethod
    def get_dataset_header():
        """Header para la sección de dataset"""
        return """
        <div class="main-header">
            <h2 style="margin: 0 0 0.5rem 0; font-size: var(--font-size-3xl);">📊 Explorador del Dataset</h2>
            <p style="margin: 0; opacity: 0.9;">Visualiza y analiza los datos de entrenamiento</p>
        </div>
        """

    @staticmethod
    def get_section_title(title, subtitle=None):
        """Título de sección estándar"""
        subtitle_html = f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""
        return f"""
        <div style="margin: var(--spacing-xl) 0 var(--spacing-md) 0;">
            <h2 class="section-title">{title}</h2>
            {subtitle_html}
        </div>
        """

    @staticmethod
    def create_result_card(categoria, confianza, modelo, tiempo, evaluation_info=None):
        """Card de resultado de predicción con información de evaluación dinámica"""
        from assets.config.colors import LiverpoolColors

        confidence_color = LiverpoolColors.get_confidence_color(confianza)

        # Información de evaluación dinámica
        evaluation_html = ""
        if evaluation_info:
            modelos_evaluados = evaluation_info.get('modelos_evaluados', 1)
            resumen = evaluation_info.get('resumen_confianzas', {})

            evaluation_html = f"""
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(236, 0, 140, 0.05); border-radius: 8px; border-left: 4px solid var(--liverpool-rosa);">
                <h5 style="margin: 0 0 0.5rem 0; color: var(--liverpool-rosa); font-size: 0.9rem;">🎯 Evaluación Dinámica</h5>
                <p style="margin: 0; font-size: 0.8rem; color: var(--liverpool-gris-oscuro);">
                    <strong>Modelos evaluados:</strong> {modelos_evaluados}/3 disponibles<br>
            """

            # Mostrar confianzas de todos los modelos
            model_names = {'tfidf': 'TF-IDF', 'sentence_transformer': 'Sentence-T', 'gemma': 'Gemma LLM'}
            for model_key, confidence_val in resumen.items():
                if confidence_val is not None:
                    model_display = model_names.get(model_key, model_key)
                    winner_mark = "🏆" if model_key == modelo.lower() else "📊"
                    evaluation_html += f"<strong>{winner_mark} {model_display}:</strong> {confidence_val:.3f} | "

            evaluation_html = evaluation_html.rstrip(" | ") + "</p></div>"

        return f"""
        <div class="result-card">
            <div class="result-header">
                <h3 class="result-title">{categoria}</h3>
                <div class="confidence-badge" style="background-color: {confidence_color};">
                    {confianza:.1%}
                </div>
            </div>
            <div class="result-metrics">
                <div class="metric">
                    <span class="metric-label">Modelo Ganador</span>
                    <span class="metric-value">{modelo.upper()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Tiempo</span>
                    <span class="metric-value">{tiempo:.1f} ms</span>
                </div>
            </div>
            {evaluation_html}
        </div>
        """

    @staticmethod
    def create_top3_card(top_3_data, alternatives_data=None):
        """Card para top 3 categorías con información de modelos alternativos"""
        if not top_3_data:
            return "<div class='empty-state'>No hay datos disponibles</div>"

        # Top 3 categorías
        items_html = ""
        for i, (cat, prob) in enumerate(list(top_3_data.items())[:3], 1):
            items_html += f"""
            <div class="top3-item">
                <div class="rank">{i}</div>
                <div class="category">{cat}</div>
                <div class="probability">{prob:.1%}</div>
            </div>
            """

        # Información de modelos alternativos
        alternatives_html = ""
        if alternatives_data:
            alternatives_html = """
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(83, 86, 90, 0.05); border-radius: 8px;">
                <h5 style="margin: 0 0 0.5rem 0; color: var(--liverpool-gris-oscuro); font-size: 0.9rem;">🤖 Otros Modelos</h5>
            """

            model_names = {'tfidf': 'TF-IDF', 'sentence_transformer': 'Sentence-T', 'gemma': 'Gemma'}
            for model_key, alt_data in alternatives_data.items():
                model_display = model_names.get(model_key, model_key)
                categoria_alt = alt_data.get('categoria', 'N/A')
                confianza_alt = alt_data.get('confianza', 0)
                tiempo_alt = alt_data.get('tiempo_ms', 0)

                alternatives_html += f"""
                <div style="font-size: 0.8rem; color: var(--liverpool-gris-oscuro); margin-bottom: 0.25rem;">
                    <strong>{model_display}:</strong> {categoria_alt} ({confianza_alt:.1%}) - {tiempo_alt:.1f}ms
                </div>
                """

            alternatives_html += "</div>"

        return f"""
        <div class="top3-card">
            <h4 class="top3-title">🏆 Top 3 Categorías</h4>
            <div class="top3-list">
                {items_html}
            </div>
            {alternatives_html}
        </div>
        """

    @staticmethod
    def create_error_card(message, details=None):
        """Card de error"""
        details_html = ""
        if details:
            details_html = f"""
            <div style="margin-top: var(--spacing-md); font-size: var(--font-size-sm); color: var(--liverpool-gris-oscuro);">
                {details}
            </div>
            """

        return f"""
        <div class="error-card">
            <div class="error-icon">⚠️</div>
            <div class="error-message">{message}</div>
            {details_html}
        </div>
        """

    @staticmethod
    def create_stats_grid(total_noticias, total_categorias, periodo="2012-2022", archivo="data.json"):
        """Grid de estadísticas del dataset"""
        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_noticias:,}</div>
                <div class="stat-label">Total Noticias</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_categorias}</div>
                <div class="stat-label">Categorías Únicas</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{periodo}</div>
                <div class="stat-label">Período</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{archivo}</div>
                <div class="stat-label">Archivo Fuente</div>
            </div>
        </div>
        """

    @staticmethod
    def create_categories_grid(top_categories_data, category_mapper):
        """Grid de categorías principales"""
        categories_html = "<div class='categories-grid'>"

        for cat, count, percentage in top_categories_data:
            emoji = category_mapper.get_emoji(cat)
            categories_html += f"""
            <div class="category-item">
                <div class="category-emoji">{emoji}</div>
                <div class="category-info">
                    <div class="category-name">{cat}</div>
                    <div class="category-stats">{count:,} noticias ({percentage:.1f}%)</div>
                </div>
            </div>
            """

        categories_html += "</div>"
        return categories_html

    @staticmethod
    def create_data_table(sample_data, category_mapper, total_noticias):
        """Tabla de datos de muestra"""
        table_html = """
        <div class="data-table">
            <div class="table-header">
                <div class="th">ID</div>
                <div class="th">Titular</div>
                <div class="th">Categoría</div>
            </div>
        """

        for idx, (headline, category) in enumerate(sample_data, 1):
            emoji = category_mapper.get_emoji(category)
            headline_truncated = (headline[:100] + '...') if len(headline) > 100 else headline

            table_html += f"""
            <div class="table-row">
                <div class="td" style="text-align: center; font-weight: 600; color: var(--liverpool-rosa);">{idx}</div>
                <div class="td headline-cell">{headline_truncated}</div>
                <div class="td category-cell">{emoji} {category}</div>
            </div>
            """

        table_html += f"""
        </div>
        <div style="text-align: center; margin-top: var(--spacing-md); color: var(--liverpool-gris-oscuro); font-size: var(--font-size-sm);">
            📋 Mostrando {len(sample_data)} noticias aleatorias de {total_noticias:,} total
        </div>
        """

        return table_html

    @staticmethod
    def create_status_display(status_type, message):
        """Display de estado del sistema"""
        status_classes = {
            'success': 'status-success',
            'error': 'status-error',
            'warning': 'status-warning',
            'info': 'status-info'
        }

        icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }

        css_class = status_classes.get(status_type, 'status-info')
        icon = icons.get(status_type, 'ℹ️')

        return f"""
        <div class="{css_class}">
            {icon} {message}
        </div>
        """

    @staticmethod
    def get_footer():
        """Footer de la aplicación"""
        return """
        <div class="liverpool-footer">
            <h3 class="footer-title">🎯 Sistema de Clasificación con Evaluación Dinámica</h3>
            <p class="footer-subtitle"><strong>Modelos:</strong> TF-IDF • Sentence-Transformers • Gemma LLM • Ensemble Dinámico</p>
            <p class="footer-description">⚡ Evalúa múltiples modelos en tiempo real • 🏆 Selecciona automáticamente el mejor • 📊 Métricas transparentes</p>
        </div>
        """

    @staticmethod
    def create_loading_state(message="Cargando..."):
        """Estado de carga"""
        return f"""
        <div class="loading-state">
            🔄 {message}
        </div>
        """

    @staticmethod
    def create_empty_state(message="No hay datos disponibles"):
        """Estado vacío"""
        return f"""
        <div class="empty-state">
            📭 {message}
        </div>
        """