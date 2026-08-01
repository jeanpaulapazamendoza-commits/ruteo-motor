"""
API del motor de ruteo.

Expone el mismo motor que usa la app de Streamlit (extraído a `motor_ruteo/`)
para que el frontend Next.js lo consuma. No reimplementa nada: importa las
funciones ya probadas.
"""
from __future__ import annotations

import math
import os
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from motor_ruteo import (
    asignar_flota,
    calcular_ruta_ors,
    hora_a_segundos,
    kmeans_balanceado,
    links_google_maps,
    proyectar_metros,
    puntos_en_poligono,
    rutear_cluster_ortools,
    segundos_a_hora,
)
from sklearn.cluster import KMeans, MiniBatchKMeans

app = FastAPI(
    title="Motor de ruteo RuteoTiendas",
    version="1.0",
    description="Agrupamiento por capacidad/flota y ruteo óptimo con OR-Tools.",
)

# El frontend corre en otro origen (Vercel / localhost:3005)
ORIGENES = os.getenv("ORIGENES_PERMITIDOS", "http://localhost:3005,http://127.0.0.1:3005")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ORIGENES.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TIENDAS = 5000


# ===========================================================
# Modelos de entrada
# ===========================================================
class Tienda(BaseModel):
    id: str
    codigo: str = ""
    nombre: str = ""
    distrito: str | None = None
    lat: float
    lon: float
    bultos: int = 1
    prioridad: int = 0
    ventana_ini: str | None = None
    ventana_fin: str | None = None


class TipoVehiculo(BaseModel):
    cantidad: int = Field(ge=1)
    capacidad: int = Field(gt=0, description="Bultos que carga")
    max_tiendas: int = Field(0, ge=0, description="0 = sin límite")


class PeticionAgrupar(BaseModel):
    tiendas: list[Tienda]
    modo: Literal["capacidad", "clasico"] = "capacidad"
    criterio: Literal["tiendas", "bultos", "flota"] = "tiendas"
    capacidad: int = 25
    k: int = 8
    flota: list[TipoVehiculo] = []
    vueltas: int = Field(1, ge=1, le=3)
    uso_flota: Literal["compacto", "minimo"] = "compacto"
    # Selección manual: lista de grupos, cada uno con ids de tienda
    grupos_manuales: list[list[str]] | None = None


class PeticionRutear(BaseModel):
    tiendas: list[Tienda]
    grupos: list[list[str]] = Field(description="Ids de tienda por grupo")
    cd_lat: float
    cd_lon: float
    motor: Literal["osrm", "haversine", "ors"] = "osrm"
    cerrado: bool = True
    tiempo_tsp: int = Field(2, ge=1, le=15)
    hora_salida: str = "08:00"
    servicio_min: float = 10
    servicio_min_bulto: float = 0
    jornada_h: float = 8
    costo_fijo: float = 0
    costo_km: float = 0
    api_key_ors: str | None = None


class PeticionSector(BaseModel):
    tiendas: list[Tienda]
    poligono: list[list[float]] = Field(description="[[lon,lat], ...]")


# ===========================================================
# Utilidades
# ===========================================================
def _texto(v) -> str | None:
    """
    Devuelve texto, o None si no hay valor.

    Los campos opcionales vacíos entran como None y pandas los puede volver
    NaN. Un NaN suelto en la respuesta rompe la serialización JSON (no es
    JSON válido) y el despacho entero se cae con un 500, así que ningún
    valor de texto sale de aquí sin pasar por esta función.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def _sin_nan(obj):
    """
    Cambia por None cualquier NaN o infinito que quede en la respuesta.

    Es la última red: un solo NaN perdido en un campo hace que FastAPI no
    pueda serializar y devuelva 500 sin decir por qué, tirando abajo el
    despacho completo por un dato opcional que venía vacío.
    """
    if isinstance(obj, dict):
        return {k: _sin_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sin_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _a_dataframe(tiendas: list[Tienda]) -> pd.DataFrame:
    if not tiendas:
        raise HTTPException(400, "No se recibió ninguna tienda.")
    if len(tiendas) > MAX_TIENDAS:
        raise HTTPException(400, f"Máximo {MAX_TIENDAS} tiendas por petición.")
    df = pd.DataFrame([t.model_dump() for t in tiendas])
    df["cantidad_bultos"] = df["bultos"].astype(float).clip(lower=0)
    df["name_sucursal"] = df["nombre"]
    df["codigo_sucursal"] = df["codigo"]
    df["latitud"] = df["lat"]
    df["longitud"] = df["lon"]
    df["ventana_ini_s"] = df["ventana_ini"].map(hora_a_segundos)
    df["ventana_fin_s"] = df["ventana_fin"].map(hora_a_segundos)
    return df


def _expandir_flota(flota: list[TipoVehiculo]) -> tuple:
    lista = []
    for t in flota:
        lista.extend([(t.capacidad, t.max_tiendas)] * t.cantidad)
    return tuple(lista)


# ===========================================================
# Endpoints
# ===========================================================
@app.get("/salud")
def salud():
    return {"estado": "ok", "motor": "OR-Tools + OSRM"}


@app.post("/agrupar")
def agrupar(p: PeticionAgrupar):
    """Agrupa las tiendas en zonas de reparto."""
    df = _a_dataframe(p.tiendas)
    Xm = proyectar_metros(df["latitud"].values, df["longitud"].values)
    prio = df["prioridad"].values.astype(int)
    ids = df["id"].tolist()

    info_flota: dict = {}
    sobrecupo = False

    if p.grupos_manuales is not None:
        indice = {tid: i for i, tid in enumerate(ids)}
        labels = np.full(len(df), -1, dtype=int)
        k = 0
        for grupo in p.grupos_manuales:
            reales = [indice[t] for t in grupo if t in indice]
            if not reales:
                continue
            for pos in reales:
                labels[pos] = k
            k += 1

    elif p.modo == "clasico":
        k = max(1, min(p.k, len(df) - 1)) if len(df) > 1 else 1
        if k == 1:
            labels = np.zeros(len(df), dtype=int)
        else:
            Modelo = MiniBatchKMeans if len(df) > 3000 else KMeans
            labels = Modelo(n_clusters=k, random_state=42, n_init=10).fit_predict(Xm)

    elif p.criterio == "flota":
        if not p.flota:
            raise HTTPException(400, "Define al menos un tipo de vehículo.")
        flota = _expandir_flota(p.flota)
        pesos = df["cantidad_bultos"].values.astype(float)
        labels, info_flota = asignar_flota(
            Xm, pesos, prio, flota, p.vueltas,
            usar_holgura=(p.uso_flota == "compacto"),
        )

    else:
        pesos = (
            df["cantidad_bultos"].values.astype(float)
            if p.criterio == "bultos"
            else np.ones(len(df))
        )
        if p.capacidad <= 0:
            raise HTTPException(400, "La capacidad debe ser mayor que cero.")
        k = max(1, math.ceil(float(pesos.sum()) / p.capacidad))
        if k == 1:
            labels = np.zeros(len(df), dtype=int)
        else:
            for _ in range(6):
                labels, sobrecupo = kmeans_balanceado(Xm, k, p.capacidad, pesos=pesos)
                if not sobrecupo:
                    break
                k = min(k + 1, len(df))

    df["grupo"] = labels
    grupos = []
    for g in sorted(x for x in set(labels) if x >= 0):
        sub = df[df["grupo"] == g]
        detalle = info_flota.get(g, {}) if info_flota else {}
        grupos.append({
            "indice": int(g),
            "tiendas": sub["id"].tolist(),
            "n_tiendas": int(len(sub)),
            "bultos": int(sub["cantidad_bultos"].sum()),
            "prioritarias": int((sub["prioridad"] > 0).sum()),
            "capacidad_vehiculo": detalle.get("capacidad"),
            "max_tiendas_vehiculo": detalle.get("max_tiendas"),
            "vuelta": detalle.get("vuelta", 1),
            "centro": {
                "lat": float(sub["latitud"].mean()),
                "lon": float(sub["longitud"].mean()),
            },
        })

    sin_asignar = df.loc[df["grupo"] < 0, "id"].tolist()
    return _sin_nan({
        "grupos": grupos,
        "sin_asignar": sin_asignar,
        "sobrecupo": bool(sobrecupo),
        "resumen": {
            "tiendas": int(len(df)),
            "grupos": len(grupos),
            "bultos": int(df["cantidad_bultos"].sum()),
            "sin_asignar": len(sin_asignar),
        },
    })


@app.post("/rutear")
def rutear(p: PeticionRutear):
    """Calcula la ruta óptima de cada grupo con OR-Tools."""
    df = _a_dataframe(p.tiendas).set_index("id", drop=False)
    salida_s = hora_a_segundos(p.hora_salida) or 8 * 3600
    jornada_s = int(p.jornada_h * 3600)

    rutas = []
    for i, ids in enumerate(p.grupos):
        presentes = [t for t in ids if t in df.index]
        if not presentes:
            continue
        sub = df.loc[presentes].copy()

        servicio = (
            p.servicio_min * 60
            + sub["cantidad_bultos"].values.astype(float) * p.servicio_min_bulto * 60
        )
        ventanas = [
            (int(a), int(b)) if pd.notna(a) and pd.notna(b) and b > a else None
            for a, b in zip(sub["ventana_ini_s"], sub["ventana_fin_s"])
        ]

        try:
            if p.motor == "ors":
                if not p.api_key_ors:
                    raise HTTPException(400, "Falta la API key de OpenRouteService.")
                coords = [(float(r.longitud), float(r.latitud), idx)
                          for idx, r in sub.iterrows()]
                res = calcular_ruta_ors(
                    [p.cd_lon, p.cd_lat], coords, p.api_key_ors,
                    cerrada=p.cerrado,
                    servicio_map={idx: int(s) for idx, s in zip(sub.index, servicio)},
                    salida_cd_s=salida_s,
                )
            else:
                res = rutear_cluster_ortools(
                    sub, p.cd_lat, p.cd_lon,
                    motor=p.motor, cerrado=p.cerrado, tiempo_tsp=p.tiempo_tsp,
                    servicio_s=servicio, ventanas=ventanas,
                    salida_cd_s=salida_s, jornada_max_s=jornada_s,
                )
        except HTTPException:
            raise
        except Exception as e:  # un grupo que falla no debe tumbar todo el despacho
            rutas.append({
                "indice": i, "error": f"{type(e).__name__}: {e}",
                "orden": [], "km": 0, "duracion_min": 0,
            })
            continue

        if not res:
            continue

        etas = res.get("etas", {}) or {}
        paradas = [{
            "id": tid,
            "orden": n + 1,
            "codigo": _texto(sub.loc[tid, "codigo_sucursal"]) or "",
            "nombre": _texto(sub.loc[tid, "name_sucursal"]) or "",
            "distrito": _texto(sub.loc[tid, "distrito"]),
            "bultos": int(sub.loc[tid, "cantidad_bultos"]),
            "prioridad": int(sub.loc[tid, "prioridad"]),
            "lat": float(sub.loc[tid, "latitud"]),
            "lon": float(sub.loc[tid, "longitud"]),
            "eta": segundos_a_hora(etas.get(tid)),
        } for n, tid in enumerate(res["orden"])]

        coords_orden = ([(p.cd_lat, p.cd_lon)]
                        + [(x["lat"], x["lon"]) for x in paradas]
                        + ([(p.cd_lat, p.cd_lon)] if p.cerrado else []))

        rutas.append({
            "indice": i,
            "paradas": paradas,
            "orden": [x["id"] for x in paradas],
            "km": round(res["distance_km"], 2),
            "duracion_min": round(res["duration_min"], 1),
            "bultos": int(sub["cantidad_bultos"].sum()),
            "costo": round(p.costo_fijo + res["distance_km"] * p.costo_km, 2),
            "salida": segundos_a_hora(res.get("salida_s")),
            "fin": segundos_a_hora(res.get("fin_s")),
            "motor": res.get("motor"),
            "aviso": res.get("aviso"),
            "geometria": [[float(la), float(lo)] for la, lo in res["coords_route"]],
            "links_maps": links_google_maps(coords_orden),
        })

    validas = [r for r in rutas if not r.get("error")]
    return _sin_nan({
        "rutas": rutas,
        "totales": {
            "rutas": len(validas),
            "km": round(sum(r["km"] for r in validas), 2),
            "duracion_min": round(sum(r["duracion_min"] for r in validas), 1),
            "costo": round(sum(r.get("costo", 0) for r in validas), 2),
            "paradas": sum(len(r.get("paradas", [])) for r in validas),
        },
    })


@app.post("/tiendas-en-sector")
def tiendas_en_sector(p: PeticionSector):
    """Qué tiendas caen dentro de un polígono dibujado en el mapa."""
    df = _a_dataframe(p.tiendas)
    dentro = puntos_en_poligono(
        df["latitud"].values, df["longitud"].values,
        [(float(x[0]), float(x[1])) for x in p.poligono],
    )
    return {"ids": df.loc[np.asarray(dentro, dtype=bool), "id"].tolist()}
