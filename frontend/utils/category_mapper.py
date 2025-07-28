"""
Mapeo de categorías con emojis y descripciones
"""

import random
from typing import Dict, List, Tuple


class CategoryMapper:
    """Mapeador de categorías con emojis y descripciones"""

    def __init__(self):
        # Mapeo principal de categorías con emojis específicos
        self.category_emojis = {
            # Noticias y Política
            'POLITICS': '🏛️',
            'U.S. NEWS': '🇺🇸',
            'WORLD NEWS': '🌍',
            'WORLDPOST': '🌐',
            'THE WORLDPOST': '📰',
            'MEDIA': '📺',
            'CRIME': '👮‍♀️',

            # Negocios y Tecnología
            'BUSINESS': '💼',
            'TECH': '💻',
            'MONEY': '💰',
            'SCIENCE': '🔬',

            # Entretenimiento
            'ENTERTAINMENT': '🎬',
            'COMEDY': '😂',
            'ARTS': '🎨',
            'ARTS & CULTURE': '🎭',
            'CULTURE & ARTS': '🖼️',
            'TASTE': '👅',

            # Deportes
            'SPORTS': '⚽',

            # Estilo de Vida
            'STYLE': '👗',
            'STYLE & BEAUTY': '💄',
            'WELLNESS': '🧘‍♀️',
            'HEALTHY LIVING': '🏃‍♀️',
            'FITNESS': '💪',
            'FOOD & DRINK': '🍽️',
            'TRAVEL': '✈️',
            'HOME & LIVING': '🏠',

            # Familia y Educación
            'PARENTING': '👶',
            'PARENTS': '👨‍👩‍👧‍👦',
            'EDUCATION': '📚',
            'COLLEGE': '🎓',
            'WEDDINGS': '💒',
            'DIVORCE': '💔',

            # Diversidad e Inclusión
            'WOMEN': '👩',
            'QUEER VOICES': '🏳️‍🌈',
            'BLACK VOICES': '✊🏿',
            'LATINO VOICES': '🌶️',

            # Medio Ambiente y Sociedad
            'GREEN': '🌱',
            'ENVIRONMENT': '🌳',
            'IMPACT': '🤝',
            'GOOD NEWS': '😊',
            'RELIGION': '⛪',
            'WEIRD NEWS': '🤪',
            'FIFTY': '5️⃣0️⃣',
        }

        # Emojis de respaldo para categorías no mapeadas
        self.backup_emojis = [
            '📰', '📝', '📊', '🔍', '💭', '📢', '🗞️', '📈',
            '🎯', '⭐', '🔥', '💡', '🚀', '✨', '🎪', '🎊'
        ]

        # Descripciones amigables
        self.category_descriptions = {
            'POLITICS': 'Política y Gobierno',
            'U.S. NEWS': 'Noticias de Estados Unidos',
            'WORLD NEWS': 'Noticias Internacionales',
            'BUSINESS': 'Negocios y Economía',
            'TECH': 'Tecnología',
            'ENTERTAINMENT': 'Entretenimiento',
            'COMEDY': 'Humor y Comedia',
            'SPORTS': 'Deportes',
            'WELLNESS': 'Bienestar y Salud',
            'STYLE & BEAUTY': 'Moda y Belleza',
            'PARENTING': 'Paternidad',
            'FOOD & DRINK': 'Comida y Bebidas',
            'TRAVEL': 'Viajes',
            'GREEN': 'Medio Ambiente',
            'SCIENCE': 'Ciencia',
            'ARTS': 'Arte y Cultura',
            'EDUCATION': 'Educación',
            'WOMEN': 'Mujeres',
            'HOME & LIVING': 'Hogar y Estilo de Vida'
        }

        # Colores para diferentes tipos de categorías
        self.category_colors = {
            # Noticias - Azul
            'POLITICS': '#1f77b4', 'U.S. NEWS': '#1f77b4', 'WORLD NEWS': '#1f77b4',
            'WORLDPOST': '#1f77b4', 'THE WORLDPOST': '#1f77b4', 'MEDIA': '#1f77b4',

            # Negocios - Verde
            'BUSINESS': '#2ca02c', 'TECH': '#2ca02c', 'MONEY': '#2ca02c', 'SCIENCE': '#2ca02c',

            # Entretenimiento - Morado
            'ENTERTAINMENT': '#9467bd', 'COMEDY': '#9467bd', 'ARTS': '#9467bd',
            'ARTS & CULTURE': '#9467bd', 'CULTURE & ARTS': '#9467bd',

            # Deportes - Naranja
            'SPORTS': '#ff7f0e',

            # Estilo de vida - Rosa
            'STYLE': '#e377c2', 'STYLE & BEAUTY': '#e377c2', 'WELLNESS': '#e377c2',
            'HEALTHY LIVING': '#e377c2', 'FOOD & DRINK': '#e377c2', 'TRAVEL': '#e377c2',

            # Familia - Amarillo
            'PARENTING': '#bcbd22', 'PARENTS': '#bcbd22', 'EDUCATION': '#bcbd22',
            'COLLEGE': '#bcbd22', 'WEDDINGS': '#bcbd22',

            # Social - Cian
            'WOMEN': '#17becf', 'QUEER VOICES': '#17becf', 'BLACK VOICES': '#17becf',
            'LATINO VOICES': '#17becf', 'IMPACT': '#17becf',

            # Medio ambiente - Verde claro
            'GREEN': '#7f7f7f', 'ENVIRONMENT': '#7f7f7f'
        }

    def get_emoji(self, category: str) -> str:
        """
        Obtiene emoji para una categoría

        Args:
            category (str): Nombre de la categoría

        Returns:
            str: Emoji correspondiente
        """
        if category in self.category_emojis:
            return self.category_emojis[category]
        else:
            # Emoji aleatorio pero consistente para categorías no mapeadas
            random.seed(hash(category))
            return random.choice(self.backup_emojis)

    def get_description(self, category: str) -> str:
        """
        Obtiene descripción amigable para una categoría

        Args:
            category (str): Nombre de la categoría

        Returns:
            str: Descripción en español
        """
        if category in self.category_descriptions:
            return self.category_descriptions[category]
        else:
            # Formatear nombre de categoría
            return category.replace('_', ' ').title()

    def get_color(self, category: str) -> str:
        """
        Obtiene color para una categoría

        Args:
            category (str): Nombre de la categoría

        Returns:
            str: Color hexadecimal
        """
        return self.category_colors.get(category, '#8c564b')  # Marrón por defecto

    def format_category(self, category: str) -> str:
        """
        Formatea categoría con emoji y descripción

        Args:
            category (str): Nombre de la categoría

        Returns:
            str: Categoría formateada
        """
        emoji = self.get_emoji(category)
        description = self.get_description(category)
        return f"{emoji} {description}"

    def format_result(self, prediction_result: Dict) -> Dict:
        """
        Formatea el resultado de predicción con emojis

        Args:
            prediction_result (Dict): Resultado de la API

        Returns:
            Dict: Resultado formateado
        """
        if 'error' in prediction_result and prediction_result['error']:
            return prediction_result

        # Categoría principal
        categoria = prediction_result['prediccion']['categoria']
        categoria_formateada = self.format_category(categoria)

        # Top 3 probabilidades formateadas
        top_3_formateadas = {}
        if 'top_3_probabilidades' in prediction_result:
            for cat, prob in prediction_result['top_3_probabilidades'].items():
                cat_formateada = self.format_category(cat)
                top_3_formateadas[cat_formateada] = prob

        # Resultado formateado
        resultado_formateado = prediction_result.copy()
        resultado_formateado['prediccion']['categoria_formateada'] = categoria_formateada
        resultado_formateado['prediccion']['emoji'] = self.get_emoji(categoria)
        resultado_formateado['prediccion']['color'] = self.get_color(categoria)
        resultado_formateado['top_3_formateadas'] = top_3_formateadas

        return resultado_formateado

    def get_all_categories_formatted(self) -> List[Tuple[str, str]]:
        """
        Obtiene todas las categorías formateadas

        Returns:
            List[Tuple[str, str]]: Lista de (categoria_original, categoria_formateada)
        """
        categorias = []
        for category in sorted(self.category_emojis.keys()):
            formatted = self.format_category(category)
            categorias.append((category, formatted))
        return categorias


# Instancia global del mapper
category_mapper = CategoryMapper()