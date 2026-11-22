"""Distancias y tiempos por calles reales, sin el tope de 100 puntos.

El servidor de calles (OSRM) no limita cuántos puntos se le mandan, sino
cuántas celdas devuelve de una vez: fuentes × destinos. El público admite
10 000 por consulta. Una matriz de 300 puntos tiene 90 000, así que no cabe en
una, pero sí en diez: se piden las filas por tandas y se juntan.

Con eso el servidor público alcanza para unos 400 puntos. Más allá no es que
no se pueda, es que tarda: pide ir a una consulta por segundo, y el planificador
tiene que responder antes de que la web corte la espera. Para eso está el
servidor propio (carpeta `osrm/`): sin pausa entre consultas y con el tope de
celdas que se le quiera dar.

Se configura con variables de entorno, sin tocar código:

  OSRM_URL            dirección del servidor propio. Vacía: el público.
  OSRM_MAX_TABLA      lado de la tabla mayor que acepta (su --max-table-size).
  OSRM_MAX_COORDS     cuántos puntos caben en la dirección de una consulta.
  OSRM_PAUSA_S        segundos de espera entre consultas.
  OSRM_MAX_CONSULTAS  cuántas consultas se aceptan para una sola matriz.

Este módulo también hace que el ruteo de siempre use el mismo servidor: las
funciones de `ruteo.py` se extraen de la app de Streamlit y no se editan a
mano, así que se les cambia la dirección desde aquí, al importar.
"""

import os
import time
from functools import lru_cache

import numpy as np
import requests

from . import ruteo
from .geo import matriz_haversine

OSRM_PUBLICO = "https://router.project-osrm.org"
OSRM_URL = (os.environ.get("OSRM_URL") or OSRM_PUBLICO).strip().rstrip("/")
ES_PROPIO = OSRM_URL != OSRM_PUBLICO


def _numero(nombre, defecto, tipo=int):
    try:
        return tipo(os.environ[nombre])
    except (KeyError, ValueError):
        return defecto


OSRM_MAX_TABLA = max(1, _numero("OSRM_MAX_TABLA", 100))
# La dirección de la consulta lleva todos los puntos escritos; pasados unos
# 8 000 caracteres el servidor público la rechaza. A 20 por punto, 350 caben.
OSRM_MAX_COORDS = max(2, _numero("OSRM_MAX_COORDS", 350))
# El servidor público pide no pasar de una consulta por segundo.
OSRM_PAUSA_S = max(0.0, _numero("OSRM_PAUSA_S", 0.0 if ES_PROPIO else 1.0, float))
# Cada consulta al público cuesta unos dos segundos con la pausa. Dieciséis
# son medio minuto: lo que se puede gastar antes de empezar a calcular.
OSRM_MAX_CONSULTAS = max(1, _numero("OSRM_MAX_CONSULTAS", 400 if ES_PROPIO else 16))

FACTOR_RODEO = 1.4


def _tandas(n):
    """Cómo trocear una matriz de n × n: (consultas, filas, columnas por consulta).

    Dos límites a la vez: filas × columnas no pasa de las celdas que el
    servidor devuelve, y los puntos escritos en la dirección no pasan de los
    que caben. Cuando van todas las columnas, las filas ya están entre ellas y
    no ocupan sitio aparte. Entre lo que cumple, lo que menos consultas pida."""
    celdas = OSRM_MAX_TABLA * OSRM_MAX_TABLA
    mejor = None
    for c in range(min(n, OSRM_MAX_COORDS), 0, -1):
        f = min(n, celdas // c)
        if c < n:
            f = min(f, OSRM_MAX_COORDS - c)
        if f < 1:
            continue
        consultas = -(-n // f) * -(-n // c)
        if mejor is None or consultas < mejor[0]:
            mejor = (consultas, f, c)
    return mejor


def consultas_necesarias(n):
    """Cuántas consultas cuesta la matriz de n puntos."""
    return _tandas(n)[0] if n > 0 else 0


def cabe_por_calles(n):
    """Si la matriz de n puntos se puede pedir sin pasarse del tiempo."""
    return consultas_necesarias(n) <= OSRM_MAX_CONSULTAS


def max_puntos_por_calles():
    """El mayor número de puntos que todavía se resuelve por calles."""
    n = 2
    while n < 20_000 and cabe_por_calles(n + 1):
        n += 1
    return n


def _pedir(url, params):
    """Una consulta, con un segundo intento si el servidor está ocupado."""
    for intento in (1, 2):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code in (429, 502, 503, 504) and intento == 1:
            time.sleep(2.0)
            continue
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"OSRM table: HTTP {r.status_code}")
        if r.status_code >= 400 or data.get("code") != "Ok":
            raise RuntimeError(f"OSRM table: {data.get('message') or data.get('code')}")
        return data
    raise RuntimeError("OSRM table: sin respuesta")


@lru_cache(maxsize=4)
def _tabla(coords_lonlat):
    n = len(coords_lonlat)
    dist = np.full((n, n), np.nan)
    dur = np.full((n, n), np.nan)
    _, f, c = _tandas(n)
    primera = True
    for c0 in range(0, n, c):
        cols = list(range(c0, min(c0 + c, n)))
        for f0 in range(0, n, f):
            filas = list(range(f0, min(f0 + f, n)))
            params = {"annotations": "distance,duration"}
            if len(cols) == n:
                # Van todos los puntos como destino; las filas son parte de ellos.
                puntos = cols
                if len(filas) < n:
                    params["sources"] = ";".join(map(str, filas))
            else:
                puntos = list(dict.fromkeys(filas + cols))
                pos = {p: k for k, p in enumerate(puntos)}
                params["sources"] = ";".join(str(pos[p]) for p in filas)
                params["destinations"] = ";".join(str(pos[p]) for p in cols)
            locs = ";".join(
                f"{coords_lonlat[p][0]:.5f},{coords_lonlat[p][1]:.5f}" for p in puntos)
            if not primera and OSRM_PAUSA_S:
                time.sleep(OSRM_PAUSA_S)
            primera = False
            data = _pedir(f"{OSRM_URL}/table/v1/driving/{locs}", params)
            sub = np.ix_(filas, cols)
            dist[sub] = np.array(data["distances"], dtype=float)
            dur[sub] = np.array(data["durations"], dtype=float)

    # Un par sin ruta (una isla, una calle cortada en el mapa) vuelve vacío:
    # se estima en línea recta por el factor de rodeo.
    if np.isnan(dist).any() or np.isnan(dur).any():
        latlon = np.array([(lat, lon) for lon, lat in coords_lonlat])
        recta = matriz_haversine(latlon) * FACTOR_RODEO
        dist = np.where(np.isnan(dist), recta, dist)
        dur = np.where(np.isnan(dur), recta / (ruteo.VEL_PROMEDIO_KMH / 3.6), dur)
    return dist, dur


def tabla_calles(coords_lonlat):
    """Matriz de distancias (m) y duraciones (s) por calles, del tamaño que sea.

    coords_lonlat: tupla de (lon, lat). Lanza una excepción si la matriz pide
    más consultas de las aceptadas o si el servidor falla; quien llama decide
    entonces si estima en línea recta."""
    coords = tuple((float(lon), float(lat)) for lon, lat in coords_lonlat)
    if not cabe_por_calles(len(coords)):
        raise RuntimeError(
            f"OSRM table: {len(coords)} puntos piden {consultas_necesarias(len(coords))} "
            f"consultas y el tope es {OSRM_MAX_CONSULTAS}.")
    dist, dur = _tabla(coords)
    return dist.copy(), dur.copy()


# El ruteo de siempre, con el mismo servidor y el mismo tope.
ruteo.OSRM_BASE = OSRM_URL
ruteo.OSRM_MAX_PUNTOS = max_puntos_por_calles()
ruteo.osrm_tabla = tabla_calles

__all__ = [
    "ES_PROPIO", "tabla_calles", "cabe_por_calles", "consultas_necesarias",
    "max_puntos_por_calles",
]
