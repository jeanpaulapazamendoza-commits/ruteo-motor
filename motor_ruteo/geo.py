"""Extraído de app.py (Streamlit) — no editar a mano sin re-extraer."""


import numpy as np

R_TIERRA = 6_371_008.8



def proyectar_metros(lats, lons):
    """Proyección equirectangular local: convierte lat/lon a coordenadas (x, y)
    en METROS. Así 1 unidad = 1 metro en cualquier dirección y el clustering
    agrupa por distancia geográfica real (StandardScaler distorsionaba esto)."""
    lat0 = np.radians(np.mean(lats))
    x = np.radians(np.asarray(lons, dtype=float)) * R_TIERRA * np.cos(lat0)
    y = np.radians(np.asarray(lats, dtype=float)) * R_TIERRA
    return np.column_stack([x, y])


def matriz_haversine(coords_latlon):
    """Matriz n x n de distancias haversine en metros (vectorizada).
    coords_latlon: array (n, 2) con columnas [latitud, longitud]."""
    lat = np.radians(coords_latlon[:, 0])[:, None]
    lon = np.radians(coords_latlon[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * R_TIERRA * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def puntos_en_poligono(lats, lons, poligono_lonlat):
    """Ray casting vectorizado: True por cada punto (lat, lon) dentro del
    polígono. ``poligono_lonlat`` viene del GeoJSON de Leaflet.draw como
    [[lon, lat], ...] (anillo exterior)."""
    poly = np.asarray(poligono_lonlat, dtype=float)
    px, py = poly[:, 0], poly[:, 1]                     # lon, lat de vértices
    x = np.asarray(lons, dtype=float)
    y = np.asarray(lats, dtype=float)
    dentro = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cruza = ((py[i] > y) != (py[j] > y)) & (
            x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-300) + px[i])
        dentro ^= cruza
        j = i
    return dentro
