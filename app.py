"""
Método de Bisección - API Flask
================================
Servidor web para cálculo numérico de raíces mediante el método de bisección.
Incluye normalización inteligente de expresiones matemáticas y criterios de decisión.
"""

from pathlib import Path
from math import isfinite
import logging

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sympy as sp
import re


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
x = sp.Symbol("x")

ALLOWED_FUNCTIONS = {
    "x": x,
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "sen": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "tg": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
}

CHARACTER_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "'": "'",
    "'": "'",
    """: '"',
    """: '"',
    "×": "*",
    "·": "*",
    "÷": "/",
}

SPANISH_FUNCTION_REPLACEMENTS = {
    "sen(": "sin(",
    "tg(": "tan(",
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def normalize_characters(text: str) -> str:
    """Reemplaza caracteres Unicode especiales por sus equivalentes ASCII."""
    for special_char, replacement in CHARACTER_REPLACEMENTS.items():
        text = text.replace(special_char, replacement)
    return text


def normalize_operators(text: str) -> str:
    """Convierte operadores matemáticos comunes a formato Python."""
    text = text.replace("^", "**")
    for spanish_func, english_func in SPANISH_FUNCTION_REPLACEMENTS.items():
        text = text.replace(spanish_func, english_func)
    return text


def add_implicit_multiplications(text: str) -> str:
    """Inserta multiplicaciones explícitas donde faltan."""
    try:
        text = re.sub(r"(\d)\s*\(", r"\1*(", text)
        text = re.sub(r"\)\s*(\d)", r")*\1", text)
        text = re.sub(r"\)\s*\(", r")*(", text)
        text = re.sub(r"(?<!sin)(?<!cos)(?<!tan)(?<!log)(?<!exp)(?<!sqrt)(?<!ln)([a-zA-Z])\s*\(", r"\1*(", text)
        text = re.sub(r"\)\s*([a-zA-Z])", r")*\1", text)
        text = re.sub(r"(\d)\s*(?<![a-zA-Z])([a-zA-Z])", r"\1*\2", text)
        return text
    except Exception as e:
        logger.error(f"Error en add_implicit_multiplications: {e}")
        return text


def fix_exponents(text: str) -> str:
    """Corrige exponentes con signos negativos o variables."""
    try:
        def fix_exponent(match: re.Match) -> str:
            exponent = match.group(1)
            needs_parentheses = any(char in exponent for char in ["x", "+", "-", "*", "/"])
            if needs_parentheses:
                exponent = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", exponent)
                return f"**({exponent})"
            return f"**{exponent}"

        text = re.sub(r"\*\*([^\s\*\/\+\-\(\)]+)", fix_exponent, text)
        return text
    except Exception as e:
        logger.error(f"Error en fix_exponents: {e}")
        return text


def prepare_expression(text: str) -> sp.Expr:
    """Convierte una expresión en texto a una expresión válida de SymPy."""
    logger.info(f"Procesando expresión: {text}")

    text = text.strip()
    if not text:
        raise ValueError("La función está vacía.")

    text = normalize_characters(text)
    text = normalize_operators(text)
    text = add_implicit_multiplications(text)
    text = fix_exponents(text)

    try:
        expression = sp.sympify(text, locals=ALLOWED_FUNCTIONS, rational=False)
        expression = sp.expand(sp.sympify(str(expression), locals=ALLOWED_FUNCTIONS, rational=False))
    except Exception as error:
        logger.error(f"Error al convertir a SymPy: {error}")
        raise ValueError(f"No se pudo interpretar la función: {error}")

    symbols = expression.free_symbols
    invalid_symbols = symbols - {x}

    if invalid_symbols:
        names = ", ".join(str(symbol) for symbol in invalid_symbols)
        raise ValueError(f"Variable(s) no permitida(s): {names}. Utiliza únicamente x.")

    logger.info(f"Expresión procesada exitosamente: {expression}")
    return expression


def evaluate(function: sp.Expr, value: float) -> float:
    """Evalúa una función SymPy en un punto específico."""
    try:
        result = function.subs(x, value)
        result = float(sp.N(result))

        if not isfinite(result):
            raise ValueError("Resultado no finito.")

        return result
    except Exception as e:
        logger.error(f"Error al evaluar en x={value}: {e}")
        raise ValueError(f"No se pudo evaluar la función en x = {value}.")


# ============================================================================
# MÉTODO DE BISECCIÓN
# ============================================================================

def calculate_bisection(
    expression: str,
    lower_limit: float,
    upper_limit: float,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """Aplica el método de bisección para encontrar una raíz."""
    logger.info(f"Iniciando bisección con: {expression}, [{lower_limit}, {upper_limit}]")

    function = prepare_expression(expression)

    lower_limit = float(lower_limit)
    upper_limit = float(upper_limit)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    if lower_limit == upper_limit:
        raise ValueError("xl y xu no pueden ser iguales.")

    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    if max_iterations <= 0:
        raise ValueError("Las iteraciones deben ser mayores que 0.")

    if tolerance <= 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")

    f_lower = evaluate(function, lower_limit)
    f_upper = evaluate(function, upper_limit)

    if f_lower == 0:
        return {
            "funcion": str(function),
            "raiz": lower_limit,
            "intervalo_final": [lower_limit, lower_limit],
            "iteraciones_realizadas": 0,
            "detenido_por": "f(xl) = 0",
            "tabla": [],
        }

    if f_upper == 0:
        return {
            "funcion": str(function),
            "raiz": upper_limit,
            "intervalo_final": [upper_limit, upper_limit],
            "iteraciones_realizadas": 0,
            "detenido_por": "f(xu) = 0",
            "tabla": [],
        }

    if f_lower * f_upper > 0:
        raise ValueError("No existe cambio de signo en el intervalo. Se necesita f(xl)·f(xu) < 0.")

    iteration_table = []
    previous_root = None
    current_root = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."

    for iteration in range(1, max_iterations + 1):
        current_root = (lower_limit + upper_limit) / 2
        f_current = evaluate(function, current_root)

        # 1. Determinar signo de f(xr)
        if f_current > 0:
            signo_fxr = "positivo (+)"
        elif f_current < 0:
            signo_fxr = "negativo (-)"
        else:
            signo_fxr = "cero (0)"

        # 2. Calcular errores
        if previous_root is None:
            absolute_error = None
            relative_error = None
            relative_error_pct = None
        else:
            absolute_error = abs(current_root - previous_root)
            if current_root != 0:
                relative_error = absolute_error / abs(current_root)
                relative_error_pct = relative_error * 100
            else:
                relative_error = None
                relative_error_pct = None

        # 3. Determinar criterio de actualización y su tipo para el frontend
        if f_current == 0:
            criterio = "f(xr) = 0 → Raíz exacta"
            criterio_tipo = "raiz_exacta"
        elif f_lower * f_current < 0:
            criterio = f"f(xr) {signo_fxr} → xu = xr"
            criterio_tipo = "xu_igual_xr"
        else:
            criterio = f"f(xr) {signo_fxr} → xl = xr"
            criterio_tipo = "xl_igual_xr"

        # 4. Registrar iteración
        iteration_row = {
            "iteracion": iteration,
            "xl": lower_limit,
            "xu": upper_limit,
            "xr": current_root,
            "fxr": f_current,
            "ea": absolute_error,
            "er": relative_error,
            "er_pct": relative_error_pct,
            "criterio": criterio,
            "criterio_tipo": criterio_tipo,
        }

        iteration_table.append(iteration_row)

        # 5. Verificar condiciones de parada
        if f_current == 0:
            stop_reason = "f(xr) = 0"
            break

        if absolute_error is not None and absolute_error <= tolerance:
            iteration_table[-1]["criterio"] += " | Tolerancia alcanzada"
            stop_reason = "Se alcanzó la tolerancia."
            break

        # 6. Actualizar intervalo
        if f_lower * f_current < 0:
            upper_limit = current_root
            f_upper = f_current
        else:
            lower_limit = current_root
            f_lower = f_current

        previous_root = current_root

    logger.info(f"Bisección completada en {len(iteration_table)} iteraciones")

    return {
        "funcion": str(function),
        "raiz": current_root,
        "intervalo_final": [lower_limit, upper_limit],
        "iteraciones_realizadas": len(iteration_table),
        "detenido_por": stop_reason,
        "tabla": iteration_table,
    }


# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.route("/")
def index():
    """Sirve la página principal de la aplicación."""
    return send_file(BASE_DIR / "index.html")


@app.route("/api/calcular", methods=["POST"])
def calcular():
    """Endpoint para calcular la bisección de una función."""
    try:
        data = request.get_json()
        logger.info(f"Recibida petición /api/calcular: {data}")

        function = data.get("funcion", "")
        lower_limit = data.get("xl")
        upper_limit = data.get("xu")
        max_iterations = data.get("iteraciones", 50)
        tolerance = data.get("tolerancia", 0.000001)

        result = calculate_bisection(function, lower_limit, upper_limit, max_iterations, tolerance)
        return jsonify({"ok": True, **result})

    except Exception as error:
        logger.error(f"Error en /api/calcular: {error}")
        return jsonify({"ok": False, "error": str(error)}), 400


@app.route("/api/puntos", methods=["POST"])
def puntos():
    """Endpoint para obtener puntos de evaluación de una función."""
    try:
        data = request.get_json()
        logger.info(f"Recibida petición /api/puntos: {data}")

        expression = data.get("funcion", "")
        min_x = float(data.get("xmin"))
        max_x = float(data.get("xmax"))

        function = prepare_expression(expression)
        num_points = 700
        points = []

        for i in range(num_points + 1):
            x_value = min_x + (max_x - min_x) * i / num_points
            try:
                y_value = evaluate(function, x_value)
                if abs(y_value) > 1e6:
                    points.append({"x": x_value, "y": None})
                else:
                    points.append({"x": x_value, "y": y_value})
            except Exception:
                points.append({"x": x_value, "y": None})

        return jsonify({"ok": True, "puntos": points})

    except Exception as error:
        logger.error(f"Error en /api/puntos: {error}")
        return jsonify({"ok": False, "error": str(error)}), 400

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    
    print("\n======================================")
    print("   MÉTODO DE BISECCIÓN - PYTHON")
    print("======================================")
    print(f"Servidor: http://127.0.0.1:{port}")
    print("======================================\n")

    app.run(host="0.0.0.0", port=port, debug=False)
