"""Planificación de toda la flota en un solo modelo.

A diferencia del resto del paquete, este módulo NO viene de la app de
Streamlit: se escribió aquí y se edita aquí.

Qué cambia respecto a `/agrupar` + `/rutear`
--------------------------------------------
Aquellos dos pasos deciden por separado: primero se reparten las tiendas por
cercanía en línea recta, sin mirar ninguna hora, y después se ordena cada grupo
por su cuenta. Si un grupo no puede cumplir sus ventanas horarias ya es tarde
para arreglarlo, porque la tienda no puede cambiarse de vehículo: el ruteo se
rendía y calculaba la ruta sin ventanas.

Aquí se decide todo a la vez, en un único modelo de OR-Tools: qué vehículo
lleva cada tienda y en qué orden, cumpliendo en el mismo cálculo

  * la capacidad en bultos de cada vehículo,
  * su máximo de paradas,
  * las ventanas horarias y el tiempo de servicio,
  * la jornada máxima, que deja de ser un aviso y pasa a ser un límite.

Lo que no cabe no se fuerza: queda sin asignar, y se dice por qué.

La agrupación de siempre no se tira. Se calcula igual y se entrega al modelo
como punto de partida; el solver la mejora desde ahí en vez de empezar de cero,
que es lo que le permite dar algo bueno en pocos segundos.
"""

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .clustering import asignar_flota
from .geo import matriz_haversine, proyectar_metros
from .ruteo import OSRM_MAX_PUNTOS, VEL_PROMEDIO_KMH, osrm_tabla

# Cuánto «cuesta» dejar una tienda sin visitar, en metros de recorrido. Tiene
# que ser mucho mayor que cualquier rodeo razonable, o el solver preferiría
# abandonar una tienda lejana antes que ir a por ella.
PENALIZACION_M = 5_000_000
# Una tienda prioritaria es la última que se abandona.
FACTOR_PRIORIDAD = 4

# Lo que cuesta sacar un vehículo más, cuando no se han dado costes reales:
# el equivalente a 15 km. Sirve para que no reparta la carga entre toda la
# flota si con menos vehículos se hace lo mismo.
COSTE_VEHICULO_M = 15_000

# Las calles no son rectas: factor de rodeo urbano sobre la distancia en línea
# recta, el mismo que ya usa el ruteo cuando OSRM no devuelve un tramo.
FACTOR_RODEO = 1.4

MOTIVO_CAPACIDAD = "Su pedido no cabe ni en el vehículo más grande de la flota."
MOTIVO_VENTANA = "Su ventana horaria cierra antes de que un vehículo pueda llegar desde el centro."
MOTIVO_VENTANA_TARDE = "Su ventana horaria abre cuando la jornada ya terminó."
MOTIVO_JORNADA = "Ir, atenderla y volver no cabe en la jornada ni con un vehículo para ella sola."
MOTIVO_FLOTA = "La flota no alcanza: la carga total supera la capacidad de todos los vehículos."
MOTIVO_SIN_HUECO = "No cupo en ninguna ruta sin pasarse de la jornada, de la capacidad o de una ventana horaria."


def _matrices(coords_latlon, motor):
    """Distancias (m) y duraciones (s) entre todos los puntos; el 0 es el centro.

    Por calles reales mientras quepa en una sola consulta al OSRM público. Con
    más puntos no se puede pedir la matriz entera, y se estima: línea recta por
    el factor de rodeo, a la velocidad media urbana. Devuelve también cuál de
    las dos se usó, para poder decirlo."""
    n = len(coords_latlon)
    if motor == "osrm" and n <= OSRM_MAX_PUNTOS:
        try:
            lonlat = tuple((float(lon), float(lat)) for lat, lon in coords_latlon)
            dist, dur = osrm_tabla(lonlat)
            return np.asarray(dist, dtype=float), np.asarray(dur, dtype=float), "calles"
        except Exception:
            pass  # el servidor público falla a veces: mejor estimar que no planificar
    dist = matriz_haversine(np.asarray(coords_latlon, dtype=float)) * FACTOR_RODEO
    dur = dist / (VEL_PROMEDIO_KMH / 3.6)
    return dist, dur, "recta"


def _descartar_imposibles(dist, dur, demanda, servicio, ventanas, caps,
                          salida_s, jornada_s, cerrado):
    """Las tiendas que ninguna planificación podría atender, con su motivo.

    Se apartan antes de construir el modelo por dos razones: el modelo queda
    más pequeño, y el motivo se puede decir con exactitud. Si se dejaran
    dentro, el solver las abandonaría igual pero sin explicar nada."""
    motivos = {}
    cap_max = max(caps)
    fin_jornada = salida_s + jornada_s if jornada_s > 0 else None
    for i in range(1, len(dist)):
        t = i - 1
        if demanda[t] > cap_max:
            motivos[t] = MOTIVO_CAPACIDAD
            continue
        llegada = salida_s + dur[0][i]
        v = ventanas[t]
        if v is not None:
            if v[1] < llegada:
                motivos[t] = MOTIVO_VENTANA
                continue
            if fin_jornada is not None and v[0] > fin_jornada:
                motivos[t] = MOTIVO_VENTANA_TARDE
                continue
            llegada = max(llegada, v[0])
        if fin_jornada is not None:
            vuelta = dur[i][0] if cerrado else 0.0
            if llegada + servicio[t] + vuelta > fin_jornada:
                motivos[t] = MOTIVO_JORNADA
    return motivos


def _rutas_iniciales(coords_tiendas, demanda, prioridad, flota, dist):
    """El punto de partida: la agrupación de siempre, convertida en rutas.

    Devuelve una lista con una ruta por vehículo (posiciones de tienda, en el
    orden del vecino más cercano). Un vehículo sin grupo lleva la lista vacía.
    Si la agrupación falla por lo que sea, no hay punto de partida y el solver
    empieza de cero: es más lento, no incorrecto."""
    try:
        Xm = proyectar_metros(coords_tiendas[:, 0], coords_tiendas[:, 1])
        labels, info = asignar_flota(
            Xm, np.asarray(demanda, dtype=float), np.asarray(prioridad, dtype=int),
            tuple(flota), 1, usar_holgura=True,
        )
    except Exception:
        return None

    libres = list(range(len(flota)))
    rutas = [[] for _ in flota]
    for r, datos in info.items():
        tipo = (int(datos["capacidad"]), int(datos["max_tiendas"]))
        # un vehículo de ese mismo tipo que todavía no tenga grupo
        v = next((x for x in libres if (int(flota[x][0]), int(flota[x][1])) == tipo), None)
        if v is None:
            continue
        libres.remove(v)
        pendientes = [int(i) for i in np.where(labels == r)[0]]
        actual, orden = 0, []
        while pendientes:
            sig = min(pendientes, key=lambda t: dist[actual][t + 1])
            orden.append(sig)
            pendientes.remove(sig)
            actual = sig + 1
        rutas[v] = orden
    return rutas


def planificar_flota(coords_tiendas, demanda, servicio, ventanas, prioridad, flota,
                     cd_lat, cd_lon, *, motor="osrm", cerrado=True, salida_s=8 * 3600,
                     jornada_s=8 * 3600, tiempo_seg=20, costo_fijo=0.0, costo_km=0.0):
    """Reparte y ordena todas las tiendas entre la flota en un solo cálculo.

    coords_tiendas: array (n, 2) de [lat, lon].
    demanda, servicio, prioridad: una entrada por tienda (bultos, segundos, nivel).
    ventanas: una entrada por tienda, `(ini_s, fin_s)` o None.
    flota: lista de `(capacidad_bultos, max_tiendas)`, una por vehículo; 0 = sin límite.

    Devuelve un dict con:
      rutas        lista de {vehiculo, orden, llegadas, fin_s, dist_m}
                   (`orden` son posiciones de tienda; `llegadas` va por posición)
      sin_asignar  {posición de tienda: motivo}
      matriz       'calles' o 'recta'
      desde_agrupacion  si el solver partió de la agrupación de siempre
    """
    coords_tiendas = np.asarray(coords_tiendas, dtype=float).reshape(-1, 2)
    n = len(coords_tiendas)
    flota = [(int(c), int(m)) for c, m in flota]
    demanda = [int(round(float(d))) for d in demanda]
    servicio = [float(s) for s in servicio]
    prioridad = [int(p) for p in prioridad]

    coords = np.vstack([[cd_lat, cd_lon], coords_tiendas])
    dist, dur, matriz = _matrices(coords, motor)

    caps = [c for c, _ in flota]
    motivos = _descartar_imposibles(dist, dur, demanda, servicio, ventanas, caps,
                                    salida_s, jornada_s, cerrado)
    vivas = [t for t in range(n) if t not in motivos]
    if not vivas:
        return {"rutas": [], "sin_asignar": motivos, "matriz": matriz,
                "desde_agrupacion": False}

    # ── El modelo trabaja sobre las tiendas posibles ───────────────────────
    # Nodo 0 = centro; nodos 1..m = tiendas vivas; nodo m+1 = final fantasma,
    # solo si la ruta es abierta (no vuelve al centro).
    m = len(vivas)
    V = len(flota)
    sub = [0] + [t + 1 for t in vivas]
    D = dist[np.ix_(sub, sub)]
    T = dur[np.ix_(sub, sub)]
    serv = [0.0] + [servicio[t] for t in vivas]
    dem = [0] + [demanda[t] for t in vivas]
    tam = m + 1

    if not cerrado:
        D = np.pad(D, ((0, 1), (0, 1)))   # llegar al final fantasma no cuesta
        T = np.pad(T, ((0, 1), (0, 1)))
        serv.append(0.0)
        dem.append(0)
        tam += 1
        gestor = pywrapcp.RoutingIndexManager(tam, V, [0] * V, [m + 1] * V)
    else:
        gestor = pywrapcp.RoutingIndexManager(tam, V, 0)
    modelo = pywrapcp.RoutingModel(gestor)

    # Distancia: es lo que se minimiza.
    D_int = [[int(round(x)) for x in fila] for fila in D]
    cb_dist = modelo.RegisterTransitMatrix(D_int)
    modelo.SetArcCostEvaluatorOfAllVehicles(cb_dist)

    # Sacar un vehículo más tiene un coste: con los costes reales si se dieron.
    if costo_km > 0 and costo_fijo > 0:
        coste_vehiculo = int(round(costo_fijo / costo_km * 1000))
    else:
        coste_vehiculo = COSTE_VEHICULO_M
    modelo.SetFixedCostOfAllVehicles(coste_vehiculo)

    # Capacidad en bultos, la de cada vehículo.
    cb_carga = modelo.RegisterUnaryTransitVector(dem)
    modelo.AddDimensionWithVehicleCapacity(cb_carga, 0, caps, True, "Carga")

    # Máximo de paradas por vehículo.
    una = [0] + [1] * m + ([0] if not cerrado else [])
    cb_paradas = modelo.RegisterUnaryTransitVector(una)
    modelo.AddDimensionWithVehicleCapacity(
        cb_paradas, 0, [mt if mt > 0 else m for _, mt in flota], True, "Paradas")

    # Tiempo: viajar de A a B cuesta el trayecto más lo que se tardó en atender A.
    T_int = [[int(round(T[a][b] + serv[a])) for b in range(tam)] for a in range(tam)]
    cb_tiempo = modelo.RegisterTransitMatrix(T_int)
    duracion_max = int(jornada_s) if jornada_s > 0 else 24 * 3600
    fin_max = int(salida_s) + duracion_max
    # La holgura es la espera: un vehículo puede llegar antes de que abra una
    # ventana y quedarse esperando, pero nunca más de lo que dura la jornada.
    modelo.AddDimension(cb_tiempo, duracion_max, fin_max, False, "Tiempo")
    tiempo = modelo.GetDimensionOrDie("Tiempo")
    for v in range(V):
        tiempo.CumulVar(modelo.Start(v)).SetRange(int(salida_s), int(salida_s))
        # La jornada es un límite, no un aviso: el vehículo tiene que haber
        # terminado —y vuelto, si la ruta es cerrada— antes de que acabe.
        tiempo.CumulVar(modelo.End(v)).SetMax(fin_max)
        modelo.AddVariableMinimizedByFinalizer(tiempo.CumulVar(modelo.End(v)))

    for pos, t in enumerate(vivas, start=1):
        idx = gestor.NodeToIndex(pos)
        v = ventanas[t]
        if v is not None:
            tiempo.CumulVar(idx).SetRange(int(max(v[0], salida_s)), int(min(v[1], fin_max)))
        # Lo que no cabe se puede dejar fuera, pagando. Sin esto, una sola
        # tienda imposible dejaría el problema entero sin solución.
        castigo = PENALIZACION_M * (FACTOR_PRIORIDAD if prioridad[t] > 0 else 1)
        modelo.AddDisjunction([idx], castigo)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(int(max(1, tiempo_seg)))

    # ── Punto de partida: la agrupación de siempre ─────────────────────────
    solucion = None
    desde_agrupacion = False
    semilla = _rutas_iniciales(
        coords_tiendas[vivas], [demanda[t] for t in vivas],
        [prioridad[t] for t in vivas], flota, D)
    if semilla is not None:
        modelo.CloseModelWithParameters(params)
        rutas_idx = [[gestor.NodeToIndex(p + 1) for p in ruta] for ruta in semilla]
        inicial = modelo.ReadAssignmentFromRoutes(rutas_idx, True)
        # La agrupación no mira horas: puede romper una ventana o la jornada.
        # En ese caso no sirve como arranque y el solver construye la suya.
        if inicial is not None:
            solucion = modelo.SolveFromAssignmentWithParameters(inicial, params)
            desde_agrupacion = solucion is not None
    if solucion is None:
        solucion = modelo.SolveWithParameters(params)
    if solucion is None:
        # No debería ocurrir —todo es descartable—, pero si ocurre se dice.
        for t in vivas:
            motivos[t] = MOTIVO_SIN_HUECO
        return {"rutas": [], "sin_asignar": motivos, "matriz": matriz,
                "desde_agrupacion": False}

    # ── Leer la solución ───────────────────────────────────────────────────
    rutas = []
    atendidas = set()
    for v in range(V):
        idx = modelo.Start(v)
        orden, llegadas, metros = [], {}, 0
        while not modelo.IsEnd(idx):
            nodo = gestor.IndexToNode(idx)
            if 1 <= nodo <= m:
                t = vivas[nodo - 1]
                orden.append(t)
                llegadas[t] = int(solucion.Value(tiempo.CumulVar(idx)))
                atendidas.add(t)
            sig = solucion.Value(modelo.NextVar(idx))
            metros += D_int[nodo][gestor.IndexToNode(sig)]
            idx = sig
        if not orden:
            continue  # vehículo que se queda en el centro
        rutas.append({
            "vehiculo": flota[v],
            "orden": orden,
            "llegadas": llegadas,
            "fin_s": int(solucion.Value(tiempo.CumulVar(idx))),
            "dist_m": float(metros),
        })

    # Lo que el solver dejó fuera, con el motivo más preciso que se puede dar.
    falta_flota = sum(demanda) > sum(caps)
    for t in vivas:
        if t not in atendidas:
            motivos[t] = MOTIVO_FLOTA if falta_flota else MOTIVO_SIN_HUECO

    return {"rutas": rutas, "sin_asignar": motivos, "matriz": matriz,
            "desde_agrupacion": desde_agrupacion}
