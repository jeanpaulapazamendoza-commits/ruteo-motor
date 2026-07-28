"""Extraído de app.py (Streamlit) — no editar a mano sin re-extraer."""


import datetime as _dt
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2



def hora_a_segundos(v):
    """Convierte una hora a segundos desde medianoche. Acepta 'HH:MM', '8:30',
    objetos time/datetime de Excel, horas decimales (8.5) o fracción de día
    de Excel (0.354). Devuelve None si está vacío o no se puede interpretar."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, _dt.time) or isinstance(v, _dt.datetime):
        return v.hour * 3600 + v.minute * 60
    if isinstance(v, str):
        try:
            partes = v.strip().replace(".", ":").split(":")
            h = int(partes[0])
            m = int(partes[1]) if len(partes) > 1 else 0
            if 0 <= h <= 24 and 0 <= m < 60:
                return h * 3600 + m * 60
        except (ValueError, IndexError):
            pass
        return None
    try:
        f = float(v)
        if 0 < f < 1:          # fracción de día (formato interno de Excel)
            return int(f * 86400)
        if 0 <= f <= 24:       # horas decimales
            return int(f * 3600)
    except (TypeError, ValueError):
        pass
    return None


def segundos_a_hora(s):
    """Segundos desde medianoche -> 'HH:MM' (o '—' si no hay dato)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "—"
    s = int(s)
    return f"{(s // 3600) % 24:02d}:{(s % 3600) // 60:02d}"


def simular_etas(secuencia, dur_s, servicio_s, salida_s):
    """Simula los horarios de una ruta siguiendo `secuencia` de nodos (0 = CD).
    Devuelve ({nodo: hora_llegada_s}, hora_fin_s). La hora de fin incluye el
    retorno al CD si la secuencia termina en 0."""
    t = float(salida_s)
    llegadas = {}
    prev = secuencia[0]
    for nodo in secuencia[1:]:
        t += float(dur_s[prev][nodo])
        if nodo != 0:
            llegadas[nodo] = t
            t += float(servicio_s[nodo])
        prev = nodo
    return llegadas, t


def resolver_tsp_tiempo(dur_s, servicio_s, ventanas, salida_s, jornada_s,
                        cerrado=True, tiempo_seg=3):
    """TSP con DIMENSIÓN DE TIEMPO en OR-Tools: respeta ventanas horarias por
    tienda, suma el tiempo de servicio y limita la jornada. Nodo 0 = CD.
    ventanas[i] = (ini_s, fin_s) o None. Devuelve (orden, llegadas, fin_s)
    o None si las ventanas son imposibles de cumplir."""
    n = len(dur_s)
    M = [[int(round(v)) for v in fila] for fila in np.asarray(dur_s)]
    serv = [int(round(s)) for s in servicio_s]
    if cerrado:
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        for fila in M:
            fila.append(0)
        M.append([0] * (n + 1))
        serv.append(0)
        manager = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [n])
    routing = pywrapcp.RoutingModel(manager)

    def transito(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return (M[a][b] if a < n and b <= n else 0) + (serv[a] if a < n else 0)

    cb = routing.RegisterTransitCallback(transito)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    horizonte = int(salida_s + (jornada_s if jornada_s > 0 else 24 * 3600))
    routing.AddDimension(cb, 24 * 3600, max(horizonte, int(salida_s) + 60), False, "Tiempo")
    dim = routing.GetDimensionOrDie("Tiempo")
    dim.CumulVar(routing.Start(0)).SetRange(int(salida_s), int(salida_s))
    for nodo in range(1, n):
        v = ventanas[nodo]
        if v is not None:
            ini, fin = int(v[0]), int(v[1])
            if fin <= ini:
                continue
            dim.CumulVar(manager.NodeToIndex(nodo)).SetRange(ini, fin)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(tiempo_seg))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    orden, llegadas = [], {}
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        nodo = manager.IndexToNode(idx)
        if nodo < n:
            orden.append(nodo)
            if nodo != 0:
                llegadas[nodo] = sol.Value(dim.CumulVar(idx))
        idx = sol.Value(routing.NextVar(idx))
    fin_s = sol.Value(dim.CumulVar(idx))
    return orden, llegadas, fin_s
