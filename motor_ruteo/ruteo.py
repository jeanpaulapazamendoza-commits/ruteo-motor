"""Extraído de app.py (Streamlit) — no editar a mano sin re-extraer."""


import time
import numpy as np
import pandas as pd
import requests
from functools import lru_cache
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from .geo import matriz_haversine, R_TIERRA
from .tiempo import simular_etas, resolver_tsp_tiempo

OSRM_BASE = 'https://router.project-osrm.org'
OSRM_MAX_PUNTOS = 100
VEL_PROMEDIO_KMH = 25



def osrm_tabla(coords_lonlat):
    """Matriz de distancias (m) y duraciones (s) por calles reales usando el
    servicio /table del servidor público de OSRM. Sin API key. Gratis.
    coords_lonlat: tupla de tuplas (lon, lat) — máx ~100 puntos por request."""
    locs = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords_lonlat)
    r = requests.get(
        f"{OSRM_BASE}/table/v1/driving/{locs}",
        params={"annotations": "distance,duration"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table: {data.get('message', data.get('code'))}")
    dist = np.array(data["distances"], dtype=float)
    dur = np.array(data["durations"], dtype=float)
    # Si algún par no tiene ruta, OSRM devuelve null -> rellenar con haversine
    if np.isnan(dist).any() or np.isnan(dur).any():
        coords_latlon = np.array([(lat, lon) for lon, lat in coords_lonlat])
        H = matriz_haversine(coords_latlon) * 1.4  # factor de rodeo urbano
        dist = np.where(np.isnan(dist), H, dist)
        dur = np.where(np.isnan(dur), H / (VEL_PROMEDIO_KMH / 3.6), dur)
    return dist, dur


def osrm_geometria(coords_lonlat_orden):
    """Geometría de la ruta por calles reales (servicio /route de OSRM público).
    Acepta cualquier cantidad de paradas: trocea en tramos de 80 y concatena.
    Devuelve lista de (lat, lon) para dibujar en el mapa."""
    coords = list(coords_lonlat_orden)
    puntos = []
    paso = 80
    tramos = [coords[i:i + paso + 1] for i in range(0, max(len(coords) - 1, 1), paso)]
    for t_idx, tramo in enumerate(tramos):
        if len(tramo) < 2:
            continue
        locs = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in tramo)
        r = requests.get(
            f"{OSRM_BASE}/route/v1/driving/{locs}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM route: {data.get('message', data.get('code'))}")
        puntos.extend((lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"])
        if t_idx < len(tramos) - 1:
            time.sleep(0.15)  # cortesía con el servidor público
    return puntos


def osrm_ruta_orden(coords_lonlat_orden):
    """Como osrm_geometria, pero además devuelve la distancia (m) y duración (s)
    reales del recorrido siguiendo ese orden exacto de paradas."""
    coords = list(coords_lonlat_orden)
    puntos, dist_total, dur_total = [], 0.0, 0.0
    paso = 80
    tramos = [coords[i:i + paso + 1] for i in range(0, max(len(coords) - 1, 1), paso)]
    for t_idx, tramo in enumerate(tramos):
        if len(tramo) < 2:
            continue
        locs = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in tramo)
        r = requests.get(
            f"{OSRM_BASE}/route/v1/driving/{locs}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM route: {data.get('message', data.get('code'))}")
        ruta = data["routes"][0]
        puntos.extend((lat, lon) for lon, lat in ruta["geometry"]["coordinates"])
        dist_total += float(ruta.get("distance", 0))
        dur_total += float(ruta.get("duration", 0))
        if t_idx < len(tramos) - 1:
            time.sleep(0.15)
    return puntos, dist_total, dur_total


def resolver_tsp(matriz_m, cerrado=True, tiempo_seg=2):
    """Resuelve el orden óptimo de visita con Google OR-Tools (ejecuta en el
    propio servidor, NO consume ninguna API). El nodo 0 es el CD.
    Para ruta abierta se agrega un nodo fantasma con costo 0 como destino.
    Devuelve la lista de nodos en orden de visita (empieza en 0 = CD)."""
    M = [[int(round(v)) for v in fila] for fila in np.asarray(matriz_m)]
    n = len(M)
    if cerrado:
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        for fila in M:
            fila.append(0)
        M.append([0] * (n + 1))
        manager = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [n])
    routing = pywrapcp.RoutingModel(manager)

    def costo(i, j):
        return M[manager.IndexToNode(i)][manager.IndexToNode(j)]

    transit = routing.RegisterTransitCallback(costo)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(tiempo_seg))
    solucion = routing.SolveWithParameters(params)
    if solucion is None:
        return None

    orden = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        nodo = manager.IndexToNode(idx)
        if nodo < n:  # excluir nodo fantasma
            orden.append(nodo)
        idx = solucion.Value(routing.NextVar(idx))
    return orden


def rutear_cluster_ortools(cluster_data, cd_lat, cd_lon, motor, cerrado, tiempo_tsp,
                           servicio_s=None, ventanas=None, salida_cd_s=8 * 3600,
                           jornada_max_s=0):
    """Calcula la ruta óptima de un cluster con OR-Tools.
    motor: 'osrm' (calles reales, gratis) o 'haversine' (línea recta, sin internet).
    servicio_s: segundos de servicio por tienda (alineado a cluster_data).
    ventanas: lista de (ini_s, fin_s) o None por tienda. Si hay ventanas, se usa
    el solver con dimensión de tiempo; si no, TSP por distancia + simulación de ETAs."""
    idx_tiendas = list(cluster_data.index)
    coords_latlon = np.vstack([
        [cd_lat, cd_lon],
        cluster_data[["latitud", "longitud"]].values.astype(float),
    ])
    n = len(coords_latlon)
    avisos = []

    usa_osrm = (motor == "osrm") and n <= OSRM_MAX_PUNTOS
    if motor == "osrm" and n > OSRM_MAX_PUNTOS:
        avisos.append(f"{n - 1} tiendas superan el límite del servidor OSRM público "
                      f"({OSRM_MAX_PUNTOS - 1}); se usó distancia haversine en este cluster.")

    if usa_osrm:
        coords_lonlat = tuple((float(lon), float(lat)) for lat, lon in coords_latlon)
        dist_m, dur_s = osrm_tabla(coords_lonlat)
    else:
        dist_m = matriz_haversine(coords_latlon)
        dur_s = dist_m / (VEL_PROMEDIO_KMH / 3.6)

    # Nodo 0 = CD; nodos 1..m = tiendas
    servicio_nodos = [0.0] + ([float(s) for s in servicio_s] if servicio_s is not None
                              else [0.0] * len(idx_tiendas))
    ventanas_nodos = [None] + (list(ventanas) if ventanas is not None
                               else [None] * len(idx_tiendas))
    tiene_ventanas = any(v is not None for v in ventanas_nodos)

    # Prioridades: las tiendas con prioridad 1 se visitan primero, luego 2, ...
    # y al final las normales (prioridad 0).
    if "prioridad" in cluster_data.columns:
        prioridades = cluster_data["prioridad"].astype(int).values
    else:
        prioridades = np.zeros(len(idx_tiendas), dtype=int)
    niveles = sorted({int(p) for p in prioridades if p > 0})

    orden_nodos = None
    llegadas, fin_s = None, None

    if tiene_ventanas:
        # Solver con dimensión de tiempo (ventanas mandan sobre prioridades)
        if niveles:
            avisos.append("Hay ventanas horarias y prioridades en el mismo grupo: "
                          "las ventanas tienen precedencia sobre el orden por prioridad.")
        rt = resolver_tsp_tiempo(dur_s, servicio_nodos, ventanas_nodos,
                                 salida_cd_s, jornada_max_s, cerrado, tiempo_tsp)
        if rt is not None:
            orden_nodos, llegadas, fin_s = rt
        else:
            avisos.append("No fue posible cumplir todas las ventanas horarias de este "
                          "grupo (revisa horas vs. hora de salida del CD); la ruta se "
                          "calculó sin ventanas.")

    if orden_nodos is None:
        # TSP por distancia (con etapas de prioridad si las hay) + simulación de ETAs
        if niveles:
            grupos = [[j + 1 for j, p in enumerate(prioridades) if int(p) == nv] for nv in niveles]
            grupos.append([j + 1 for j, p in enumerate(prioridades) if int(p) == 0])
            dist_arr = np.asarray(dist_m)
            orden_nodos = [0]
            actual = 0
            for grupo in grupos:
                if not grupo:
                    continue
                sub = [actual] + grupo
                orden_sub = resolver_tsp(dist_arr[np.ix_(sub, sub)], cerrado=False,
                                         tiempo_seg=tiempo_tsp)
                if orden_sub is None:
                    return None
                orden_nodos.extend(sub[j] for j in orden_sub[1:])
                actual = orden_nodos[-1]
        else:
            orden_nodos = resolver_tsp(dist_m, cerrado=cerrado, tiempo_seg=tiempo_tsp)
            if orden_nodos is None:
                return None
        secuencia_sim = orden_nodos + ([0] if cerrado else [])
        llegadas, fin_s = simular_etas(secuencia_sim, dur_s, servicio_nodos, salida_cd_s)
        # avisar si la simulación viola alguna ventana
        if tiene_ventanas:
            fuera = sum(1 for nodo, t in llegadas.items()
                        if ventanas_nodos[nodo] is not None
                        and not (ventanas_nodos[nodo][0] <= t <= ventanas_nodos[nodo][1]))
            if fuera:
                avisos.append(f"{fuera} tiendas quedarían fuera de su ventana horaria.")

    secuencia = orden_nodos + ([0] if cerrado else [])
    dist_total = float(sum(dist_m[a][b] for a, b in zip(secuencia[:-1], secuencia[1:])))
    dur_total = float(fin_s - salida_cd_s)
    orden_df = [idx_tiendas[nodo - 1] for nodo in orden_nodos if nodo != 0]
    etas_df = {idx_tiendas[nodo - 1]: t for nodo, t in llegadas.items() if nodo != 0}

    if jornada_max_s > 0 and dur_total > jornada_max_s:
        avisos.append(f"La ruta dura {dur_total / 3600:.1f} h y excede la jornada "
                      f"máxima de {jornada_max_s / 3600:.1f} h: divide el grupo o "
                      f"reduce la capacidad por ruta.")

    coords_orden = [tuple(coords_latlon[nodo]) for nodo in secuencia]
    if usa_osrm:
        try:
            coords_route = osrm_geometria(tuple((lon, lat) for lat, lon in coords_orden))
        except Exception:
            coords_route = coords_orden  # si falla la geometría, línea recta
    else:
        coords_route = coords_orden

    return {
        "orden": orden_df,
        "coords_route": coords_route,
        "distance_km": dist_total / 1000.0,
        "duration_min": dur_total / 60.0,
        "motor": "OR-Tools + OSRM (calles reales)" if usa_osrm else "OR-Tools + Haversine (línea recta)",
        "aviso": " | ".join(avisos) if avisos else None,
        "personalizada": False,
        "etas": etas_df,
        "salida_s": salida_cd_s,
        "fin_s": fin_s,
    }


def evaluar_orden_personalizado(cluster_data, cd_lat, cd_lon, orden_df, cerrado,
                                servicio_map=None, salida_cd_s=8 * 3600):
    """Recalcula distancia, duración, geometría y ETAs de una ruta siguiendo un
    orden de visita definido por el usuario (lista de índices del DataFrame).
    Intenta calles reales con OSRM; si falla, usa haversine."""
    servicio_map = servicio_map or {}
    paradas = [(float(cluster_data.loc[idx, "latitud"]), float(cluster_data.loc[idx, "longitud"]))
               for idx in orden_df]
    coords = [(cd_lat, cd_lon)] + paradas + ([(cd_lat, cd_lon)] if cerrado else [])
    # duraciones por tramo con haversine (para repartir las ETAs)
    H = matriz_haversine(np.array(coords))
    legs_hav = [float(H[j][j + 1]) / (VEL_PROMEDIO_KMH / 3.6) for j in range(len(coords) - 1)]
    try:
        puntos, dist_m, dur_viaje_s = osrm_ruta_orden(tuple((lon, lat) for lat, lon in coords))
        motor = "✏️ Orden personalizado · OSRM (calles reales)"
        # escalar los tramos haversine para que sumen la duración real de OSRM
        factor = dur_viaje_s / sum(legs_hav) if sum(legs_hav) > 0 else 1.0
        legs = [l * factor for l in legs_hav]
    except Exception:
        dist_m = float(sum(H[j][j + 1] for j in range(len(coords) - 1))) * 1.4
        legs = [l * 1.4 for l in legs_hav]
        puntos = coords
        motor = "✏️ Orden personalizado · Haversine (línea recta)"
    # simular ETAs con tiempo de servicio
    t = float(salida_cd_s)
    etas = {}
    for pos, idx in enumerate(orden_df):
        t += legs[pos]
        etas[idx] = t
        t += float(servicio_map.get(idx, 0))
    if cerrado:
        t += legs[-1]
    fin_s = t
    return {
        "orden": list(orden_df),
        "coords_route": puntos,
        "distance_km": dist_m / 1000.0,
        "duration_min": (fin_s - salida_cd_s) / 60.0,
        "motor": motor,
        "aviso": None,
        "personalizada": True,
        "etas": etas,
        "salida_s": salida_cd_s,
        "fin_s": fin_s,
    }


def calcular_ruta_ors(cd_coord, tiendas_coords, api_key, cerrada=True,
                      servicio_map=None, salida_cd_s=8 * 3600):
    """Motor original: API de optimización de OpenRouteService (HeiGIT).
    Límite del plan gratuito: ~50 tiendas por cluster y 500 requests/día.
    servicio_map: {idx: segundos de servicio} opcional."""
    client = ors.Client(key=api_key)
    servicio_map = servicio_map or {}
    jobs = [optimization.Job(id=int(idx), location=[float(lon), float(lat)],
                             service=int(servicio_map.get(idx, 0)))
            for (lon, lat, idx) in tiendas_coords]
    if cerrada:
        vehicle = optimization.Vehicle(id=1, profile='driving-car', start=cd_coord, end=cd_coord)
    else:
        vehicle = optimization.Vehicle(id=1, profile='driving-car', start=cd_coord)
    result = client.optimization(jobs=jobs, vehicles=[vehicle], geometry=True)
    if not result.get('routes'):
        return None
    route = result['routes'][0]
    decoded = convert.decode_polyline(route['geometry'])
    coords_route = [(lat, lon) for lon, lat in decoded['coordinates']]
    steps = route.get('steps', [])
    orden_visita = [step['id'] for step in steps if step.get('type') == 'job']
    # ETAs: VROOM entrega 'arrival' en segundos relativos al inicio del vehículo
    etas = {step['id']: salida_cd_s + step.get('arrival', 0)
            for step in steps if step.get('type') == 'job'}
    dur_total_s = route.get('duration', 0) + route.get('service', 0)
    return {
        'orden': orden_visita,
        'coords_route': coords_route,
        'distance_km': route.get('distance', 0) / 1000.0,
        'duration_min': dur_total_s / 60.0,
        'motor': 'OpenRouteService (calles reales)',
        'aviso': None,
        'personalizada': False,
        'etas': etas,
        'salida_s': salida_cd_s,
        'fin_s': salida_cd_s + dur_total_s,
    }


def links_google_maps(coords_orden):
    """Links de Google Maps con las paradas en orden de visita. Google acepta
    ~10 puntos por link, así que se generan tramos con un punto de empalme."""
    links = []
    paso = 10
    coords = list(coords_orden)
    i = 0
    while i < len(coords) - 1:
        tramo = coords[i:i + paso]
        if len(tramo) >= 2:
            path = "/".join(f"{lat:.6f},{lon:.6f}" for lat, lon in tramo)
            links.append(f"https://www.google.com/maps/dir/{path}")
        i += paso - 1
    return links


def link_waze(lat, lon):
    return f"https://waze.com/ul?ll={lat:.6f},{lon:.6f}&navigate=yes"
