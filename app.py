"""
Método de Bisección / Punto Fijo / Aitken - API Flask
====================================================
Servidor web para cálculo numérico de raíces.
Incluye normalización inteligente de expresiones matemáticas y criterios de decisión.
Métodos soportados: Bisección, Punto Fijo y Aitken (aceleración).
Lógica mejorada para ser flexible e ignorar los primeros 2 errores de evaluación.
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
    for special_char, replacement in CHARACTER_REPLACEMENTS.items():
        text = text.replace(special_char, replacement)
    return text


def normalize_operators(text: str) -> str:
    text = text.replace("^", "**")
    for spanish_func, english_func in SPANISH_FUNCTION_REPLACEMENTS.items():
        text = text.replace(spanish_func, english_func)
    return text


def add_implicit_multiplications(text: str) -> str:
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
        logger.warning(f"Intento fallido de conversión estándar: {error}. Intentando modo flexible...")
        try:
            # Modo flexible: intentar sympify sin expandir y con evaluación diferida
            expression = sp.sympify(text, locals=ALLOWED_FUNCTIONS, rational=False, evaluate=False)
        except Exception as error2:
            logger.error(f"Error definitivo al convertir a SymPy: {error2}")
            raise ValueError(f"No se pudo interpretar la función. Revisa la sintaxis (ej: usa 'sin' en lugar de 'sen', '*' para multiplicar). Detalle: {error}")

    symbols = expression.free_symbols
    invalid_symbols = symbols - {x}
    if invalid_symbols:
        names = ", ".join(str(symbol) for symbol in invalid_symbols)
        logger.warning(f"Variables no permitidas detectadas: {names}. Se intentará sustituir por 'x' automáticamente.")
        for sym in invalid_symbols:
            expression = expression.subs(sym, x)
        
        if expression.free_symbols - {x}:
            raise ValueError(f"Variable(s) no permitida(s): {names}. Utiliza únicamente 'x'.")

    logger.info(f"Expresión procesada exitosamente: {expression}")
    return expression


def evaluate(function: sp.Expr, value: float) -> float:
    try:
        result = function.subs(x, value)
        result = float(sp.N(result))
        if not isfinite(result):
            raise ValueError("Resultado no finito.")
        return result
    except Exception as e:
        logger.error(f"Error al evaluar en x={value}: {e}")
        raise ValueError(f"No se pudo evaluar la función en x = {value}.")


def robust_evaluate(function: sp.Expr, value: float, error_stats: dict) -> float:
    """
    Evalúa la función de manera robusta. Ignora los primeros 2 errores 
    intentando un punto ligeramente ajustado para evitar asíntotas o límites de dominio.
    """
    try:
        return evaluate(function, value)
    except Exception:
        error_stats['count'] += 1
        if error_stats['count'] <= 2:
            logger.warning(f"Error de evaluación ignorado ({error_stats['count']}/2) en x={value}. Ajustando punto...")
            adjusted_value = value + 1e-9 if value >= 0 else value - 1e-9
            try:
                return evaluate(function, adjusted_value)
            except Exception:
                return float('nan')
        return float('nan')


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

    error_stats = {'count': 0}
    f_lower = robust_evaluate(function, lower_limit, error_stats)
    f_upper = robust_evaluate(function, upper_limit, error_stats)

    # Ajuste flexible de límites si fallan inicialmente
    if not isfinite(f_lower):
        f_lower = robust_evaluate(function, lower_limit + 1e-6, error_stats)
        if isfinite(f_lower): lower_limit += 1e-6
        
    if not isfinite(f_upper):
        f_upper = robust_evaluate(function, upper_limit - 1e-6, error_stats)
        if isfinite(f_upper): upper_limit -= 1e-6

    if not isfinite(f_lower) or not isfinite(f_upper):
        raise ValueError("No se pudo evaluar la función en los límites del intervalo, incluso con ajustes. Revisa el dominio de la función.")

    if f_lower == 0:
        return {"funcion": str(function), "raiz": lower_limit, "intervalo_final": [lower_limit, lower_limit], "iteraciones_realizadas": 0, "detenido_por": "f(xl) = 0", "tabla": [], "metodo": "biseccion"}
    if f_upper == 0:
        return {"funcion": str(function), "raiz": upper_limit, "intervalo_final": [upper_limit, upper_limit], "iteraciones_realizadas": 0, "detenido_por": "f(xu) = 0", "tabla": [], "metodo": "biseccion"}
    if f_lower * f_upper > 0:
        raise ValueError("No existe cambio de signo en el intervalo evaluado. Asegúrate de que la función cruce el eje X entre xl y xu (f(xl)·f(xu) < 0).")

    iteration_table = []
    previous_root = None
    current_root = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."

    for iteration in range(1, max_iterations + 1):
        current_root = (lower_limit + upper_limit) / 2
        f_current = robust_evaluate(function, current_root, error_stats)

        if not isfinite(f_current):
            logger.warning(f"No se pudo evaluar en xr={current_root}. Se detiene la iteración por seguridad.")
            stop_reason = "Error de evaluación no recuperable. Se detuvo para evitar resultados inválidos."
            iteration_table.append({
                "iteracion": iteration, "xl": lower_limit, "xu": upper_limit, "xr": current_root,
                "fxr": "NaN", "ea": None, "er": None, "er_pct": None,
                "criterio": "Error de evaluación (NaN)", "criterio_tipo": "error_evaluacion"
            })
            break

        signo_fxr = "positivo (+)" if f_current > 0 else "negativo (-)" if f_current < 0 else "cero (0)"

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
                relative_error = relative_error_pct = None

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
            "iteracion": iteration, "xl": lower_limit, "xu": upper_limit, "xr": current_root,
            "fxr": f_current, "ea": absolute_error, "er": relative_error, "er_pct": relative_error_pct,
            "criterio": criterio, "criterio_tipo": criterio_tipo,
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
        "funcion": str(function), "raiz": current_root, "intervalo_final": [lower_limit, upper_limit],
        "iteraciones_realizadas": len(iteration_table), "detenido_por": stop_reason,
        "tabla": iteration_table, "metodo": "biseccion",
    }


# ============================================================================
# MÉTODO DEL PUNTO FIJO
# ============================================================================
def calculate_fixed_point(expression: str, x0: float, max_iterations: int, tolerance: float) -> dict:
    logger.info(f"Iniciando punto fijo con g(x)={expression}, x0={x0}")

    g = prepare_expression(expression)
    x0 = float(x0)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Las iteraciones y la tolerancia deben ser mayores que 0.")

    iteration_table = []
    current = x0
    previous = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."
    root = current
    error_stats = {'count': 0}

    for iteration in range(1, max_iterations + 1):
        next_val = robust_evaluate(g, current, error_stats)

        if not isfinite(next_val):
            logger.warning(f"No se pudo evaluar g(x) en x={current}. Se detiene la iteración.")
            stop_reason = "Error de evaluación no recuperable en Punto Fijo."
            iteration_table.append({
                "iteracion": iteration, "xn": current, "gxn": "NaN", "xr": "NaN",
                "ea": None, "er": None, "er_pct": None, "residual": None,
                "criterio": "Error de evaluación (NaN)", "criterio_tipo": "error_evaluacion"
            })
            break

        absolute_error = abs(next_val - current) if previous is not None or iteration > 1 else abs(next_val - current)
        relative_error = (absolute_error / abs(next_val)) if next_val != 0 and absolute_error is not None else None
        relative_error_pct = relative_error * 100 if relative_error is not None else None

        criterio = "Tolerancia alcanzada | x ← g(x)" if absolute_error <= tolerance else "x ← g(x)"

        iteration_table.append({
            "iteracion": iteration, "xn": current, "gxn": next_val, "xr": next_val,
            "ea": absolute_error, "er": relative_error, "er_pct": relative_error_pct,
            "residual": abs(next_val - current), "criterio": criterio, "criterio_tipo": "punto_fijo",
        })

        root = next_val
        if absolute_error <= tolerance:
            stop_reason = "Se alcanzó la tolerancia."
            break
        if abs(next_val) > 1e12:
            stop_reason = "Divergencia detectada (|x| muy grande)."
            break

        previous = current
        current = next_val

    return {
        "funcion": str(g), "raiz": root, "x0": x0, "iteraciones_realizadas": len(iteration_table),
        "detenido_por": stop_reason, "tabla": iteration_table, "metodo": "punto_fijo", "intervalo_final": [root, root],
    }


# ============================================================================
# MÉTODO DE AITKEN (aceleración de Δ²)
# ============================================================================
def calculate_aitken(expression: str, x0: float, max_iterations: int, tolerance: float) -> dict:
    logger.info(f"Iniciando Aitken con g(x)={expression}, x0={x0}")

    g = prepare_expression(expression)
    x0 = float(x0)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Las iteraciones y la tolerancia deben ser mayores que 0.")

    seq = [x0]
    error_stats = {'count': 0}
    
    # Generación inicial tolerante a fallos
    seq.append(robust_evaluate(g, seq[0], error_stats) if isfinite(seq[0]) else float('nan'))
    seq.append(robust_evaluate(g, seq[1], error_stats) if isfinite(seq[1]) else float('nan'))

    stop_reason = "Se alcanzó el número máximo de iteraciones."
    root = seq[-1]
    previous_aitken = None
    iteration_table = []

    for iteration in range(1, max_iterations + 1):
        x_n, x_n1, x_n2 = seq[-3], seq[-2], seq[-1]

        if not isfinite(x_n) or not isfinite(x_n1) or not isfinite(x_n2):
            stop_reason = "Error de evaluación no recuperable en la secuencia de Aitken."
            break

        delta1 = x_n1 - x_n
        delta2 = x_n2 - 2 * x_n1 + x_n

        if abs(delta2) < 1e-30:
            aitken_val = x_n2
            criterio = "Δ² ≈ 0 → se usa último término de punto fijo"
            criterio_tipo = "aitken_fallback"
        else:
            aitken_val = x_n - (delta1 ** 2) / delta2
            criterio = "Â = xₙ − (Δxₙ)² / Δ²xₙ"
            criterio_tipo = "aitken"

        absolute_error = abs(aitken_val - previous_aitken) if previous_aitken is not None else abs(aitken_val - x_n2)
        relative_error = (absolute_error / abs(aitken_val)) if aitken_val != 0 else None
        relative_error_pct = relative_error * 100 if relative_error is not None else None

        if absolute_error <= tolerance:
            criterio += " | Tolerancia alcanzada"

        iteration_table.append({
            "iteracion": iteration, "xn": x_n, "xn1": x_n1, "xn2": x_n2,
            "delta1": delta1, "delta2": delta2, "xr": aitken_val, "aitken": aitken_val,
            "ea": absolute_error, "er": relative_error, "er_pct": relative_error_pct,
            "criterio": criterio, "criterio_tipo": criterio_tipo,
        })

        root = aitken_val
        previous_aitken = aitken_val

        if absolute_error <= tolerance:
            stop_reason = "Se alcanzó la tolerancia."
            break
        if abs(aitken_val) > 1e12:
            stop_reason = "Divergencia detectada (|x| muy grande)."
            break

        next_pf = robust_evaluate(g, seq[-1], error_stats)
        if not isfinite(next_pf):
            stop_reason = "Error de evaluación no recuperable al continuar la secuencia de Aitken."
            break
        seq.append(next_pf)

    return {
        "funcion": str(g), "raiz": root, "x0": x0, "iteraciones_realizadas": len(iteration_table),
        "detenido_por": stop_reason, "tabla": iteration_table, "metodo": "aitken", "intervalo_final": [root, root],
    }


# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================
@app.route("/")
def index():
    return send_file(BASE_DIR / "index.html")


@app.route("/api/calcular", methods=["POST"])
def calcular():
    try:
        data = request.get_json()
        logger.info(f"Recibida petición /api/calcular: {data}")

        metodo = (data.get("metodo") or "biseccion").lower().strip()
        function = data.get("funcion", "")
        max_iterations = int(data.get("iteraciones", 50))
        tolerance = float(data.get("tolerancia", 0.000001))

        if metodo == "biseccion":
            lower_limit = float(data.get("xl", 0))
            upper_limit = float(data.get("xu", 1))
            result = calculate_bisection(function, lower_limit, upper_limit, max_iterations, tolerance)
        elif metodo in ("punto_fijo", "puntofijo", "fixed_point"):
            x0 = data.get("x0") if data.get("x0") is not None else data.get("xl", 0.0)
            result = calculate_fixed_point(function, float(x0), max_iterations, tolerance)
        elif metodo == "aitken":
            x0 = data.get("x0") if data.get("x0") is not None else data.get("xl", 0.0)
            result = calculate_aitken(function, float(x0), max_iterations, tolerance)
        else:
            raise ValueError(f"Método desconocido: '{metodo}'. Use: biseccion, punto_fijo o aitken.")

        return jsonify({"ok": True, **result})
    except Exception as error:
        logger.warning(f"Error controlado en /api/calcular: {error}")
        # Devuelve 200 OK con ok: False para que el frontend lo maneje como un mensaje de validación, no como error de red
        return jsonify({
            "ok": False, 
            "error": f"Ocurrió un problema: {str(error)}. Verifica que la función esté bien escrita (ej: 'sin(x)' en lugar de 'sen(x)'), y que los valores iniciales estén en el dominio de la función."
        }), 200


@app.route("/api/puntos", methods=["POST"])
def puntos():
    try:
        data = request.get_json()
        expression = data.get("funcion", "")
        min_x = float(data.get("xmin"))
        max_x = float(data.get("xmax"))

        function = prepare_expression(expression)
        num_points = 700
        points = []
        error_stats = {'count': 0}
        
        for i in range(num_points + 1):
            x_value = min_x + (max_x - min_x) * i / num_points
            try:
                y_value = robust_evaluate(function, x_value, error_stats)
                if not isfinite(y_value) or abs(y_value) > 1e6:
                    points.append({"x": x_value, "y": None})
                else:
                    points.append({"x": x_value, "y": y_value})
            except Exception:
                points.append({"x": x_value, "y": None})

        return jsonify({"ok": True, "puntos": points})
    except Exception as error:
        logger.warning(f"Error controlado en /api/puntos: {error}")
        return jsonify({"ok": False, "error": str(error)}), 200


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))

    print("\n======================================")
    print("  MÉTODOS NUMÉRICOS - PYTHON")
    print("  Bisección | Punto Fijo | Aitken")
    print("  (Modo Flexible Activado)")
    print("======================================")
    print(f"Servidor: http://127.0.0.1:{port}")
    print("======================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
