"""
Configuración de colores corporativos institucionales
"""


class LiverpoolColors:
    """Paleta de colores Liverpool México"""

    # Colores principales
    ROSA_LIVERPOOL = "#ec008c"  # Pantone 226 C - Color principal corporativo
    GRIS_OSCURO = "#53565a"  # Pantone Cool Gray 11 C - Texto y fondos secundarios
    GRIS_CLARO = "#d1d3d4"  # Pantone Cool Gray 3 C - Background neutro
    NEGRO = "#101820"  # Pantone Black C - Tipografía intensa
    BLANCO = "#ffffff"  # Fondo neutro principal

    # Variaciones para UI
    ROSA_HOVER = "#d1007a"  # Rosa más oscuro para hover
    ROSA_LIGHT = "#f5e6f0"  # Rosa claro para backgrounds
    GRIS_MEDIO = "#8a8d91"  # Gris intermedio
    GRIS_BORDER = "#e5e7eb"  # Gris para bordes

    # Estados semánticos manteniendo armonía
    SUCCESS = "#10b981"  # Verde éxito
    WARNING = "#f59e0b"  # Amarillo advertencia
    ERROR = "#ef4444"  # Rojo error
    INFO = ROSA_LIVERPOOL  # Info usa el rosa principal

    # Gradientes corporativos
    GRADIENT_PRIMARY = f"linear-gradient(135deg, {ROSA_LIVERPOOL} 0%, {ROSA_HOVER} 100%)"
    GRADIENT_HEADER = f"linear-gradient(135deg, {ROSA_LIVERPOOL} 0%, {GRIS_OSCURO} 100%)"
    GRADIENT_LIGHT = f"linear-gradient(135deg, {BLANCO} 0%, {GRIS_CLARO} 100%)"

    # Sombras con colores corporativos
    SHADOW_PRIMARY = f"0 8px 32px rgba(236, 0, 140, 0.15)"  # Sombra rosa sutil
    SHADOW_CARD = f"0 4px 6px rgba(83, 86, 90, 0.1)"  # Sombra gris
    SHADOW_HOVER = f"0 8px 25px rgba(236, 0, 140, 0.2)"  # Sombra hover rosa

    @classmethod
    def get_css_variables(cls):
        """Retorna las variables CSS para usar en estilos"""
        return f"""
        :root {{
            /* Colores principales Liverpool */
            --liverpool-rosa: {cls.ROSA_LIVERPOOL};
            --liverpool-rosa-hover: {cls.ROSA_HOVER};
            --liverpool-rosa-light: {cls.ROSA_LIGHT};
            --liverpool-gris-oscuro: {cls.GRIS_OSCURO};
            --liverpool-gris-claro: {cls.GRIS_CLARO};
            --liverpool-gris-medio: {cls.GRIS_MEDIO};
            --liverpool-negro: {cls.NEGRO};
            --liverpool-blanco: {cls.BLANCO};

            /* Estados */
            --color-success: {cls.SUCCESS};
            --color-warning: {cls.WARNING};
            --color-error: {cls.ERROR};
            --color-info: {cls.INFO};

            /* Gradientes */
            --gradient-primary: {cls.GRADIENT_PRIMARY};
            --gradient-header: {cls.GRADIENT_HEADER};
            --gradient-light: {cls.GRADIENT_LIGHT};

            /* Sombras */
            --shadow-primary: {cls.SHADOW_PRIMARY};
            --shadow-card: {cls.SHADOW_CARD};
            --shadow-hover: {cls.SHADOW_HOVER};

            /* Bordes */
            --border-color: {cls.GRIS_BORDER};
            --border-radius: 12px;
            --border-radius-large: 16px;

            /* Espaciado */
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-xxl: 3rem;

            /* Tipografía */
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            --font-size-xs: 0.75rem;
            --font-size-sm: 0.875rem;
            --font-size-base: 1rem;
            --font-size-lg: 1.125rem;
            --font-size-xl: 1.25rem;
            --font-size-2xl: 1.5rem;
            --font-size-3xl: 1.875rem;
            --font-size-4xl: 2.25rem;
        }}
        """

    @classmethod
    def get_confidence_color(cls, confidence_value):
        """Retorna color basado en nivel de confianza"""
        if confidence_value >= 0.8:
            return cls.SUCCESS
        elif confidence_value >= 0.6:
            return cls.WARNING
        elif confidence_value >= 0.4:
            return cls.ROSA_LIVERPOOL
        else:
            return cls.ERROR