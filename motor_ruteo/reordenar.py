"""Volver a ordenar lo que le queda a una sola ruta.

Como `planificar.py`, `calles.py` e `insertar.py`, este módulo NO viene de la
app de Streamlit: se escribió aquí y se edita aquí.

Para qué sirve
--------------
`/planificar` decide el día entero y `/insertar` mete pedidos sin tocar el
orden. Esto es lo de en medio: una ruta que ya existe, con las paradas que ya
tiene, y la pregunta «¿en qué orden conviene hacer lo que queda?».

Se le dice de dónde parte —el centro, o donde esté el vehículo ahora— y dónde
termina: de vuelta en el centro, en una parada concreta, o donde caiga. Las
primeras `fijas` no se mueven: el conductor ya va hacia ellas.

Sirve también para lo contrario: con `optimizar=False` no cambia nada y solo
dice cuánto recorre y a qué hora llega a cada parada el orden que se le dio.
Es lo que hace falta cuando el orden lo pone una persona a mano.

Las ventanas horarias se priorizan pero no bloquean: una ruta que ya va tarde
no tiene ningún orden que las cumpla todas, y negarse a ordenarla no ayuda a
nadie. Cada segundo de retraso cuesta mucho más que un metro de rodeo, así
que el orden elegido es el que menos tarde llega y, a igualdad, el más corto.
"""

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import numpy as np

from .planificar import _matrices

# Lo que cuesta, en metros de recorrido, cada segundo que se llega tarde a
# una ventana. A 25 km/h un vehículo hace unos 7 m por segundo: con 60, un
# minuto de retraso pesa como 3,6 km de rodeo.
TARDE_M_POR_S = 60


def _recorrer(seq, inicio, hora_s, D, T, serv, vent, fin_nodo):
    """Llegadas, hora de fin, metros y segundos de retraso de un orden."""
    t, prev, metros, tarde = float(hora_s), inicio, 0.0, 0.0
    llegadas = []
    for n in seq:
        t += T[prev][n]
        metros += D[prev][n]
        v = vent[n]
        if v is not None:
            if t < v[0]:
                t = v[0]                    # llegó antes de que abran: espera
            elif t > v[1]:
                tarde += t - v[1]
        llegadas.append(t)
        t += serv[n]
        prev = n
    if fin_nodo is not None:
        t += T[prev][fin_nodo]
        metros += D[prev][fin_nodo]
    return llegadas, t, metros, tarde


def _resumen(ids, seq, llegadas, fin, metros, vent):
    return {
        "orden": [ids[n] for n in seq],
        "llegadas": {ids[n]: int(round(t)) for n, t in zip(seq, llegadas)},
        "fin_s": int(round(fin)),
        "dist_m": float(metros),
        "tarde": [ids[n] for n, t in zip(seq, llegadas)
                  if vent[n] is not None and t > vent[n][1] + 1],
    }


def reordenar_ruta(paradas, lat, lon, hora_s, *, centro=None, final="centro",
                   final_id=None, fijas=0, motor="osrm", optimizar=True,
                   tiempo_seg=5):
    """Ordena las paradas que le quedan a una ruta.

    paradas: lista de dicts, en su orden actual, con id, lat, lon,
        servicio_s y ventana `(ini_s, fin_s)` o None.
    lat, lon, hora_s: de dónde parte el vehículo y a qué hora.
    centro: `(lat, lon)` al que vuelve si `final` es 'centro'. Sin él, vuelve
        al punto de partida.
    final: 'centro' (vuelve), 'parada' (termina en `final_id`) o 'libre'
        (termina donde le convenga).
    fijas: cuántas de las primeras no se mueven.
    optimizar: False = no cambia el orden, solo lo evalúa.

    Devuelve un dict con
        antes, despues  {orden, llegadas, fin_s, dist_m, tarde}
        cambio          si el orden nuevo es distinto del que había
        matriz          'calles' o 'recta'
    """
    n = len(paradas)
    ids = [None] + [p["id"] for p in paradas]
    if final == "parada" and final_id not in ids:
        final = "libre"
    if centro is None:
        centro = (lat, lon)

    # Nodo 0 = de dónde parte; 1..n = paradas; n+1 = el centro al que vuelve.
    coords = [(float(lat), float(lon))] + [(float(p["lat"]), float(p["lon"])) for p in paradas]
    fin_nodo = None
    if final == "centro":
        coords.append((float(centro[0]), float(centro[1])))
        fin_nodo = n + 1
    dist, dur, matriz = _matrices(np.asarray(coords, dtype=float), motor)
    D = np.asarray(dist, dtype=float).tolist()
    T = np.asarray(dur, dtype=float).tolist()
    serv = [0.0] + [float(p.get("servicio_s") or 0.0) for p in paradas] + [0.0]
    vent = [None] + [p.get("ventana") for p in paradas] + [None]

    actual = list(range(1, n + 1))
    lleg, fin, met, tarde = _recorrer(actual, 0, hora_s, D, T, serv, vent, fin_nodo)
    antes = _resumen(ids, actual, lleg, fin, met, vent)
    sin_cambio = {"antes": antes, "despues": antes, "cambio": False, "matriz": matriz}

    fijas = max(0, min(int(fijas), n))
    prefijo, movibles = actual[:fijas], actual[fijas:]
    ultimo = None
    if final == "parada":
        ultimo = ids.index(final_id)
        if ultimo in prefijo:
            ultimo = None                   # ya está fija delante: no puede ser la última
        else:
            movibles = [x for x in movibles if x != ultimo]
    if not optimizar or len(movibles) + (1 if ultimo else 0) < 2:
        # Con una sola parada libre no hay nada que ordenar; con `final` en
        # una parada, basta con ponerla al final.
        if optimizar and ultimo is not None and actual[-1] != ultimo:
            seq = prefijo + movibles + [ultimo]
            lleg, fin, met, _ = _recorrer(seq, 0, hora_s, D, T, serv, vent, fin_nodo)
            return {"antes": antes, "despues": _resumen(ids, seq, lleg, fin, met, vent),
                    "cambio": True, "matriz": matriz}
        return sin_cambio

    # ── De dónde sale lo que se puede mover: tras las fijas ────────────────
    lleg_p, _, _, _ = _recorrer(prefijo, 0, hora_s, D, T, serv, vent, None)
    salida_nodo = prefijo[-1] if prefijo else 0
    salida_s = (lleg_p[-1] + serv[salida_nodo]) if prefijo else float(hora_s)

    # ── Modelo de un vehículo ──────────────────────────────────────────────
    # Nodos locales: 0 = salida; 1..m = movibles; después, el final.
    locales = [salida_nodo] + movibles
    if ultimo is not None:
        locales.append(ultimo)
        fin_local = len(locales) - 1
    elif fin_nodo is not None:
        locales.append(fin_nodo)
        fin_local = len(locales) - 1
    else:
        locales.append(None)                # final fantasma: llegar no cuesta
        fin_local = len(locales) - 1
    tam = len(locales)

    def d(a, b):
        return 0.0 if locales[a] is None or locales[b] is None else D[locales[a]][locales[b]]

    def tt(a, b):
        if locales[a] is None or locales[b] is None:
            return 0.0
        # El servicio de la salida ya está contado en `salida_s`.
        return T[locales[a]][locales[b]] + (serv[locales[a]] if a != 0 else 0.0)

    gestor = pywrapcp.RoutingIndexManager(tam, 1, [0], [fin_local])
    modelo = pywrapcp.RoutingModel(gestor)
    D_int = [[int(round(d(a, b))) for b in range(tam)] for a in range(tam)]
    T_int = [[int(round(tt(a, b))) for b in range(tam)] for a in range(tam)]
    modelo.SetArcCostEvaluatorOfAllVehicles(modelo.RegisterTransitMatrix(D_int))
    cb_t = modelo.RegisterTransitMatrix(T_int)
    horizonte = int(salida_s) + 48 * 3600
    modelo.AddDimension(cb_t, 24 * 3600, horizonte, False, "Tiempo")
    tiempo = modelo.GetDimensionOrDie("Tiempo")
    tiempo.CumulVar(modelo.Start(0)).SetRange(int(salida_s), int(salida_s))
    for a in range(1, tam):
        nodo = locales[a]
        if nodo is None or vent[nodo] is None:
            continue
        idx = modelo.End(0) if a == fin_local else gestor.NodeToIndex(a)
        v = vent[nodo]
        # Antes de que abran se espera; después de que cierren se paga.
        tiempo.CumulVar(idx).SetMin(int(min(max(v[0], salida_s), horizonte)))
        tiempo.SetCumulVarSoftUpperBound(idx, int(max(v[1], salida_s)), TARDE_M_POR_S)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(max(1, min(int(tiempo_seg), 1 + len(movibles) // 8)))

    # Parte del orden que ya tiene: lo que salga nunca será peor.
    modelo.CloseModelWithParameters(params)
    inicial = modelo.ReadAssignmentFromRoutes(
        [[gestor.NodeToIndex(a) for a in range(1, len(movibles) + 1)]], True)
    sol = (modelo.SolveFromAssignmentWithParameters(inicial, params)
           if inicial is not None else modelo.SolveWithParameters(params))
    if sol is None:
        return sin_cambio

    nuevo = []
    idx = sol.Value(modelo.NextVar(modelo.Start(0)))
    while not modelo.IsEnd(idx):
        nuevo.append(locales[gestor.IndexToNode(idx)])
        idx = sol.Value(modelo.NextVar(idx))
    seq = prefijo + nuevo + ([ultimo] if ultimo is not None else [])
    lleg2, fin2, met2, tarde2 = _recorrer(seq, 0, hora_s, D, T, serv, vent, fin_nodo)

    # Solo se propone si de verdad mejora: menos retraso, o igual y más corto.
    coste_antes = met + TARDE_M_POR_S * tarde
    coste_despues = met2 + TARDE_M_POR_S * tarde2
    # Salvo que se haya pedido terminar en una parada que no era la última:
    # eso es una orden, y se cumple aunque el recorrido salga más largo.
    obligado = ultimo is not None and actual[-1] != ultimo
    if seq == actual or (not obligado and coste_despues >= coste_antes - 1):
        return sin_cambio
    return {"antes": antes, "despues": _resumen(ids, seq, lleg2, fin2, met2, vent),
            "cambio": True, "matriz": matriz}
