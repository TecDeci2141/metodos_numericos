"""
Método de Bisección / Punto Fijo / Aitken - API Flask
====================================================
Servidor web para cálculo numérico de raíces.
Incluye normalización inteligente de expresiones matemáticas y criterios de decisión.
Métodos soportados: Bisección, Punto Fijo y Aitken (aceleración).
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
    format="%(asctime)s - %(levelname)s - %(message)s",
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
    '"': '"',
    '"': '"',
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
        text = re.sub(
            r"(?<!sin)(?<!cos)(?<!tan)(?<!log)(?<!exp)(?<!sqrt)(?<!ln)([a-zA-Z])\s*\(",
            r"\1*(",
            text,
        )
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
        expression = sp.expand(
            sp.sympify(str(expression), locals=ALLOWED_FUNCTIONS, rational=False)
        )
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
            "metodo": "biseccion",
        }
    if f_upper == 0:
        return {
            "funcion": str(function),
            "raiz": upper_limit,
            "intervalo_final": [upper_limit, upper_limit],
            "iteraciones_realizadas": 0,
            "detenido_por": "f(xu) = 0",
            "tabla": [],
            "metodo": "biseccion",
        }
    if f_lower * f_upper > 0:
        raise ValueError(
            "No existe cambio de signo en el intervalo. Se necesita f(xl)·f(xu) < 0."
        )

    iteration_table = []
    previous_root = None
    current_root = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."

    for iteration in range(1, max_iterations + 1):
        current_root = (lower_limit + upper_limit) / 2
        f_current = evaluate(function, current_root)

        if f_current > 0:
            signo_fxr = "positivo (+)"
        elif f_current < 0:
            signo_fxr = "negativo (-)"
        else:
            signo_fxr = "cero (0)"

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

        if f_current == 0:
            criterio = "f(xr) = 0 → Raíz exacta"
            criterio_tipo = "raiz_exacta"
        elif f_lower * f_current < 0:
            criterio = f"f(xr) {signo_fxr} → xu = xr"
            criterio_tipo = "xu_igual_xr"
        else:
            criterio = f"f(xr) {signo_fxr} → xl = xr"
            criterio_tipo = "xl_igual_xr"

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

        if f_current == 0:
            stop_reason = "f(xr) = 0"
            break
        if absolute_error is not None and absolute_error <= tolerance:
            iteration_table[-1]["criterio"] += " | Tolerancia alcanzada"
            stop_reason = "Se alcanzó la tolerancia."
            break

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
        "metodo": "biseccion",
    }


# ============================================================================
# MÉTODO DEL PUNTO FIJO
# ============================================================================
def calculate_fixed_point(
    expression: str,
    x0: float,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """
    Método del Punto Fijo: x_{n+1} = g(x_n)
    La expresión debe ser g(x), es decir la función de iteración.
    """
    logger.info(f"Iniciando punto fijo con g(x)={expression}, x0={x0}")

    g = prepare_expression(expression)
    x0 = float(x0)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    if max_iterations <= 0:
        raise ValueError("Las iteraciones deben ser mayores que 0.")
    if tolerance <= 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")

    iteration_table = []
    current = x0
    previous = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."
    root = current

    for iteration in range(1, max_iterations + 1):
        try:
            next_val = evaluate(g, current)
        except Exception as e:
            raise ValueError(f"Error al evaluar g(x) en iteración {iteration}: {e}")

        # Siempre calculamos el error (para la tabla), pero lo ignoramos
        # para la decisión de parada en las primeras 2 iteraciones.
        absolute_error = abs(next_val - current)

        if next_val != 0:
            relative_error = absolute_error / abs(next_val)
            relative_error_pct = relative_error * 100
        else:
            relative_error = None
            relative_error_pct = None

        g_val = next_val
        residual = abs(next_val - current)  # |g(x) - x|

        criterio = "x ← g(x)"
        # Solo consideramos tolerancia a partir de la iteración 3
        if iteration > 2 and absolute_error <= tolerance:
            criterio = "Tolerancia alcanzada | x ← g(x)"

        iteration_row = {
            "iteracion": iteration,
            "xn": current,
            "gxn": g_val,
            "xr": next_val,  # alias para compatibilidad con frontend
            "ea": absolute_error,
            "er": relative_error,
            "er_pct": relative_error_pct,
            "residual": residual,
            "criterio": criterio,
            "criterio_tipo": "punto_fijo",
        }
        iteration_table.append(iteration_row)

        root = next_val

        # Ignorar los primeros dos errores para la condición de parada
        if iteration > 2 and absolute_error <= tolerance:
            stop_reason = "Se alcanzó la tolerancia."
            break

        # Detección de divergencia simple
        if abs(next_val) > 1e12:
            stop_reason = "Divergencia detectada (|x| muy grande)."
            break

        previous = current
        current = next_val

    logger.info(f"Punto fijo completado en {len(iteration_table)} iteraciones")
    return {
        "funcion": str(g),
        "raiz": root,
        "x0": x0,
        "iteraciones_realizadas": len(iteration_table),
        "detenido_por": stop_reason,
        "tabla": iteration_table,
        "metodo": "punto_fijo",
        "intervalo_final": [root, root],  # compatibilidad
    }


# ============================================================================
# MÉTODO DE AITKEN (aceleración de Δ²)
# ============================================================================
def calculate_aitken(
    expression: str,
    x0: float,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """
    Método de Aitken (Δ²): acelera una secuencia generada por punto fijo.
    Genera tres términos de la iteración de punto fijo y aplica la fórmula de Aitken:
        â = x_n - (Δx_n)² / Δ²x_n
    donde Δx_n = x_{n+1} - x_n ,  Δ²x_n = x_{n+2} - 2x_{n+1} + x_n
    """
    logger.info(f"Iniciando Aitken con g(x)={expression}, x0={x0}")

    g = prepare_expression(expression)
    x0 = float(x0)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    if max_iterations <= 0:
        raise ValueError("Las iteraciones deben ser mayores que 0.")
    if tolerance <= 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")

    iteration_table = []
    # Necesitamos al menos 3 términos de la secuencia de punto fijo
    seq = [x0]
    try:
        seq.append(evaluate(g, seq[0]))
        seq.append(evaluate(g, seq[1]))
    except Exception as e:
        raise ValueError(f"Error al generar secuencia inicial de punto fijo: {e}")

    stop_reason = "Se alcanzó el número máximo de iteraciones."
    root = seq[-1]
    previous_aitken = None

    for iteration in range(1, max_iterations + 1):
        # Tomamos los últimos 3 términos de la secuencia de punto fijo
        x_n = seq[-3]
        x_n1 = seq[-2]
        x_n2 = seq[-1]

        delta1 = x_n1 - x_n
        delta2 = x_n2 - 2 * x_n1 + x_n  # Δ²

        if abs(delta2) < 1e-30:
            # Evitar división por cero; usamos el último término de punto fijo
            aitken_val = x_n2
            criterio = "Δ² ≈ 0 → se usa último término de punto fijo"
            criterio_tipo = "aitken_fallback"
        else:
            aitken_val = x_n - (delta1 ** 2) / delta2
            criterio = "Â = xₙ − (Δxₙ)² / Δ²xₙ"
            criterio_tipo = "aitken"

        # Siempre calculamos el error (para la tabla)
        if previous_aitken is not None:
            absolute_error = abs(aitken_val - previous_aitken)
            if aitken_val != 0:
                relative_error = absolute_error / abs(aitken_val)
                relative_error_pct = relative_error * 100
            else:
                relative_error = None
                relative_error_pct = None
        else:
            absolute_error = abs(aitken_val - x_n2)
            relative_error = None
            relative_error_pct = None

        # Solo consideramos tolerancia a partir de la iteración 3
        if iteration > 2 and absolute_error is not None and absolute_error <= tolerance:
            criterio += " | Tolerancia alcanzada"

        iteration_row = {
            "iteracion": iteration,
            "xn": x_n,
            "xn1": x_n1,
            "xn2": x_n2,
            "delta1": delta1,
            "delta2": delta2,
            "xr": aitken_val,  # valor acelerado (compatibilidad)
            "aitken": aitken_val,
            "ea": absolute_error,
            "er": relative_error,
            "er_pct": relative_error_pct,
            "criterio": criterio,
            "criterio_tipo": criterio_tipo,
        }
        iteration_table.append(iteration_row)

        root = aitken_val
        previous_aitken = aitken_val

        # Ignorar los primeros dos errores para la condición de parada
        if iteration > 2 and absolute_error is not None and absolute_error <= tolerance:
            stop_reason = "Se alcanzó la tolerancia."
            break

        if abs(aitken_val) > 1e12:
            stop_reason = "Divergencia detectada (|x| muy grande)."
            break

        # Generar el siguiente término de la secuencia de punto fijo
        try:
            next_pf = evaluate(g, seq[-1])
            seq.append(next_pf)
        except Exception as e:
            stop_reason = f"Error al continuar la secuencia de punto fijo: {e}"
            break

    logger.info(f"Aitken completado en {len(iteration_table)} iteraciones")
    return {
        "funcion": str(g),
        "raiz": root,
        "x0": x0,
        "iteraciones_realizadas": len(iteration_table),
        "detenido_por": stop_reason,
        "tabla": iteration_table,
        "metodo": "aitken",
        "intervalo_final": [root, root],  # compatibilidad
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
    """Endpoint unificado para Bisección, Punto Fijo y Aitken."""
    try:
        data = request.get_json()
        logger.info(f"Recibida petición /api/calcular: {data}")

        metodo = (data.get("metodo") or "biseccion").lower().strip()
        function = data.get("funcion", "")
        max_iterations = data.get("iteraciones", 50)
        tolerance = data.get("tolerancia", 0.000001)

        if metodo == "biseccion":
            lower_limit = data.get("xl")
            upper_limit = data.get("xu")
            result = calculate_bisection(
                function, lower_limit, upper_limit, max_iterations, tolerance
            )
        elif metodo in ("punto_fijo", "puntofijo", "fixed_point"):
            x0 = data.get("x0")
            if x0 is None:
                # fallback: intentar usar xl como x0
                x0 = data.get("xl")
            if x0 is None:
                raise ValueError("Se requiere el valor inicial x0 para el método de Punto Fijo.")
            result = calculate_fixed_point(function, x0, max_iterations, tolerance)
        elif metodo == "aitken":
            x0 = data.get("x0")
            if x0 is None:
                x0 = data.get("xl")
            if x0 is None:
                raise ValueError("Se requiere el valor inicial x0 para el método de Aitken.")
            result = calculate_aitken(function, x0, max_iterations, tolerance)
        else:
            raise ValueError(
                f"Método desconocido: '{metodo}'. Use: biseccion, punto_fijo o aitken."
            )

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
    print("  MÉTODOS NUMÉRICOS - PYTHON")
    print("  Bisección | Punto Fijo | Aitken")
    print("======================================")
    print(f"Servidor: http://127.0.0.1:{port}")
    print("======================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
