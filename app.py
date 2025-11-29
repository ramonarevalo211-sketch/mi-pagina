from math import pi, log10
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---- Física / utilidades ----
RHO_WATER = 998.0       # kg/m3 (≈ agua 20°C)
MU_WATER = 0.001002     # Pa·s (viscosidad dinámica)
G = 9.81

def calc_f(Re, e, D):
    """Factor de fricción — ecuación de Swamee-Jain"""
    if Re < 2000:
        return 64 / Re
    return 0.25 / (math.log10(e/(3.7*D) + 5.74/(Re**0.9)) ** 2)

def area(D):
    return pi * D**2 / 4.0

def velocity_from_Q(Q, D):
    A = area(D)
    return Q / A

def reynolds_number(Q, D, rho=RHO_WATER, mu=MU_WATER):
    V = abs(velocity_from_Q(Q, D))
    Re = abs(rho * V * D / mu)
    return Re

def friction_factor_swamee_jain(Re, D, eps=0.000045):
    # Swamee-Jain: f = 0.25 / [ log10( eps/(3.7D) + 5.74/Re^0.9 ) ]^2
    if Re <= 0:
        return 1e6
    if Re < 2000:
        return 64.0 / Re
    term = (eps / (3.7 * D)) + (5.74 / (Re ** 0.9))
    # guard against log10 domain errors
    if term <= 0:
        term = 1e-12
    f = 0.25 / (log10(term) ** 2)
    return f

def head_loss(Q, D, L):
    if Q <= 0:
        return 0.0
    V = velocity_from_Q(Q, D)
    Re = reynolds_number(Q, D)
    f = friction_factor_swamee_jain(Re, D)
    hf = f * (L / D) * (V * V / (2 * G))
    return hf

# Solver for two parallel pipes: find Q1 such that hf1(Q1) == hf2(QT - Q1)
def solve_parallel(QT, D1, L1, D2, L2, tol=1e-9, max_iter=100):
    # Edge cases
    if QT <= 0 or D1 <= 0 or D2 <= 0 or L1 <= 0 or L2 <= 0:
        return {"Q1": 0.0, "Q2": 0.0, "hf": 0.0, "it": 0}

    a = 1e-12
    b = QT - 1e-12
    if b <= a:
        return {"Q1": QT, "Q2": 0.0, "hf": head_loss(QT, D1, L1), "it": 0}

    def f(Q1):
        Q2 = QT - Q1
        return head_loss(Q1, D1, L1) - head_loss(Q2, D2, L2)

    fa = f(a)
    fb = f(b)

    # try to ensure sign change; if not, return split equally
    if fa * fb > 0:
        # try shrinking b
        for _ in range(20):
            b = max(a + 1e-15, b * 0.9)
            fb = f(b)
            if fa * fb <= 0:
                break
        else:
            # fallback: return split equally
            Q1 = QT / 2.0
            return {"Q1": Q1, "Q2": QT - Q1, "hf": head_loss(Q1, D1, L1), "it": 0}

    it = 0
    while it < max_iter:
        mid = (a + b) / 2.0
        fm = f(mid)
        if abs(fm) < tol:
            break
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
        it += 1

    Q1 = (a + b) / 2.0
    Q2 = QT - Q1
    hf = head_loss(Q1, D1, L1)
    return {"Q1": Q1, "Q2": Q2, "hf": hf, "it": it}

# ---- Rutas ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reynolds', methods=['POST'])
def api_reynolds():
    data = request.json or {}
    # Expect: { "nu": 1.00e-6, "D": 0.05, "Q": 0.01 }
    try:
        nu = float(data.get('nu', MU_WATER / RHO_WATER))  # si mandan cinemática, la usan
        D = float(data.get('D', 0.0))
        Q = float(data.get('Q', 0.0))
    except Exception:
        return jsonify({"error": "Entrada inválida"}), 400

    if not (nu > 0 and D > 0 and Q != 0):
        return jsonify({"error": "Valores deben ser mayores que cero y Q distinto de 0"}), 400

    V = (4.0 * Q) / (pi * D**2)
    # Si el usuario envía nu (viscosidad cinemática), Re = V*D/nu
    Re = (V * D) / nu
    flow_type = "laminar" if Re < 2000 else ("transicional" if Re <= 4000 else "turbulento")
    return jsonify({
        "V": V,
        "Re": Re,
        "flow_type": flow_type
    })

@app.route('/api/parallel', methods=['POST'])
def api_parallel():
    data = request.json or {}
    try:
        QT = float(data.get('QT', 0.0))
        D1 = float(data.get('D1', 0.0))
        L1 = float(data.get('L1', 0.0))
        D2 = float(data.get('D2', 0.0))
        L2 = float(data.get('L2', 0.0))
    except Exception:
        return jsonify({"error": "Entrada inválida"}), 400

    if not (QT > 0 and D1 > 0 and D2 > 0 and L1 > 0 and L2 > 0):
        return jsonify({"error": "Todos los parámetros deben ser mayores que cero"}), 400

    res = solve_parallel(QT, D1, L1, D2, L2)
    return jsonify(res)

@app.post("/calcularSistema")
def calcular_sistema():
    datos = request.get_json()
    ramales = datos["ramales"]

    rho = 1000      # densidad agua
    mu  = 0.001     # viscosidad

    resultados = []

    for r in ramales:

        L = r["longitud"]
        D = r["diametro"] / 1000
        e = r["rugosidad"]
        Q = r["caudal"] / 1000

        A  = math.pi * (D/2)**2
        V  = Q / A

        Re = (rho * V * D) / mu
        f  = calc_f(Re, e, D)

        hf_mayor = f * (L/D) * (V**2/(2*9.81))

        K_total = (
            0.9 * r["codos"] +
            10  * r["globo"] +
            0.2 * r["compuerta"] +
            0.5 * r["entradas"] +
            1.0 * r["salidas"]
        )

        hf_menor = K_total * (V**2 / (2*9.81))

        resultados.append({
            "velocidad": V,
            "reynolds": Re,
            "f": f,
            "h_mayor": hf_mayor,
            "h_menor": hf_menor,
            "h_total": hf_mayor + hf_menor
        })

    return jsonify({
        "ok": True,
        "resultados": resultados
    })




if __name__ == '__main__':
    app.run(debug=True)
