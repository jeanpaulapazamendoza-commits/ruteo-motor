"""
API del motor de ruteo.

Expone el mismo motor que usa la app de Streamlit (extraído a `motor_ruteo/`)
para que el frontend Next.js lo consuma. No reimplementa nada: importa las
funciones ya probadas.
"""
from __future__ import annotations

import math
import os
import unicodedata
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
    osrm_ruta_orden,
    proyectar_metros,
    puntos_en_poligono,
    rutear_cluster_ortools,
    segundos_a_hora,
)
# Estos no salen de la app de Streamlit: se importan por su nombre para que
# volver a extraer el motor (que reescribe __init__.py) no los deje fuera.
from motor_ruteo import calles as calles_osrm
from motor_ruteo.insertar import insertar_en_rutas
from motor_ruteo.planificar import planificar_flota
from motor_ruteo.reordenar import reordenar_ruta
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
    # Lo que el pedido exige del vehículo. Solo lo mira /planificar.
    etiquetas: list[str] = []


class TipoVehiculo(BaseModel):
    cantidad: int = Field(ge=1)
    capacidad: int = Field(gt=0, description="Bultos que carga")
    max_tiendas: int = Field(0, ge=0, description="0 = sin límite")
    # Lo que tiene este tipo de vehículo: refrigerado, moto, rampa…
    etiquetas: list[str] = []
    # De qué base salen, cuando hay varias. Solo lo mira /planificar.
    base: str | None = None


class Base(BaseModel):
    """Un centro del que salen vehículos."""
    id: str
    nombre: str = ""
    lat: float
    lon: float


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


class PeticionPlanificar(BaseModel):
    """Reparto y orden en un solo cálculo. Ver `motor_ruteo/planificar.py`."""
    tiendas: list[Tienda]
    flota: list[TipoVehiculo]
    cd_lat: float
    cd_lon: float
    motor: Literal["osrm", "haversine"] = "osrm"
    cerrado: bool = True
    hora_salida: str = "08:00"
    servicio_min: float = 10
    servicio_min_bulto: float = 0
    jornada_h: float = 8
    costo_fijo: float = 0
    costo_km: float = 0
    # Segundos que se le dan al solver para mejorar. El puente de la web corta
    # a los 115, y después todavía hay que pedir el trazado de cada ruta.
    tiempo_s: int = Field(20, ge=3, le=75)
    # 0 = las prioritarias solo son las últimas en quedarse fuera; 1 = además
    # se visitan antes si el rodeo es razonable; más = con más empeño.
    peso_prioridad: float = Field(1, ge=0, le=5)
    balanceo: Literal["ninguno", "tiempo", "paradas"] = "ninguno"
    # Flota repartida en varios centros. Vacío = todos salen de cd_lat/cd_lon.
    # Un tipo de vehículo sin base sale de la primera.
    bases: list[Base] = []


class RutaEnCurso(BaseModel):
    """Una ruta tal como está ahora mismo: solo lo que le queda por hacer."""
    id: str
    # Dónde está el vehículo. Sin posición, se entiende que sigue en el centro.
    lat: float | None = None
    lon: float | None = None
    # Su centro, si no es el común: de ahí salió y ahí vuelve.
    base_lat: float | None = None
    base_lon: float | None = None
    # Desde qué hora está libre en ese punto. Vacío = la hora de la petición.
    hora: str | None = None
    # Lo que le queda, en su orden. Lo ya gestionado no se manda.
    pendientes: list[Tienda] = []
    # Cuántas de las primeras no se tocan: el conductor ya va hacia ellas.
    fijas: int = Field(0, ge=0)
    # Lo que aún admite. Vacío = sin límite.
    capacidad_libre: int | None = None
    paradas_libres: int | None = None
    # Lo que tiene el vehículo. Vacío = no se sabe: no restringe, pero se avisa.
    etiquetas: list[str] | None = None


class PeticionInsertar(BaseModel):
    """Pedidos nuevos contra rutas en marcha. Ver `motor_ruteo/insertar.py`."""
    rutas: list[RutaEnCurso]
    nuevas: list[Tienda]
    cd_lat: float
    cd_lon: float
    motor: Literal["osrm", "haversine"] = "osrm"
    cerrado: bool = True
    hora_actual: str = Field(description="HH:MM, en la zona de la operación")
    servicio_min: float = 10
    servicio_min_bulto: float = 0
    # Hora a la que tiene que haber terminado todo. Vacío = sin límite.
    fin_jornada: str | None = None
    # Retocar el orden de las rutas que reciban algo. Las demás nunca se tocan.
    reordenar: bool = False
    peso_prioridad: float = Field(1, ge=0, le=5)


class PeticionReordenar(BaseModel):
    """Lo que le queda a una ruta, para ordenarlo. Ver `motor_ruteo/reordenar.py`."""
    # Las paradas pendientes, en su orden actual.
    paradas: list[Tienda]
    # De dónde parte el vehículo y a qué hora.
    lat: float
    lon: float
    hora: str = Field(description="HH:MM, en la zona de la operación")
    # El centro al que vuelve, si `final` es 'centro'.
    centro_lat: float | None = None
    centro_lon: float | None = None
    # Dónde termina: de vuelta en el centro, en una parada, o donde caiga.
    final: Literal["centro", "parada", "libre"] = "centro"
    final_id: str | None = None
    # Cuántas de las primeras no se mueven.
    fijas: int = Field(0, ge=0)
    # Paradas que conservan su puesto: las demás se ordenan a su alrededor.
    ancladas: list[str] = []
    motor: Literal["osrm", "haversine"] = "osrm"
    servicio_min: float = 10
    servicio_min_bulto: float = 0
    # False = no cambia el orden: solo dice cuánto recorre y a qué hora llega.
    optimizar: bool = True
    tiempo_s: int = Field(5, ge=1, le=30)


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
    # Las etiquetas son listas y no caben en una celda con la que luego se
    # agrupa o se compara: quien las necesita las lee de la petición.
    df = pd.DataFrame([t.model_dump(exclude={"etiquetas"}) for t in tiendas])
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


def _etiquetas(lista) -> frozenset:
    """Etiquetas comparables: sin mayúsculas, tildes ni espacios sobrantes.

    «Refrigerado », «refrigerado» y «REFRIGERADO» tienen que ser la misma: una
    viene del archivo del cliente y la otra la escribe quien planifica."""
    limpias = set()
    for e in lista or []:
        t = unicodedata.normalize("NFKD", str(e)).encode("ascii", "ignore").decode()
        t = " ".join(t.lower().split())
        if t:
            limpias.add(t)
    return frozenset(limpias)


# ===========================================================
# Endpoints
# ===========================================================
@app.get("/salud")
def salud():
    # Qué servidor de calles está en uso y hasta cuántos puntos llega: sirve
    # para comprobar desde fuera que la configuración surtió efecto.
    return {
        "estado": "ok",
        "motor": "OR-Tools + OSRM",
        "calles": "propio" if calles_osrm.ES_PROPIO else "publico",
        "max_puntos_calles": calles_osrm.max_puntos_por_calles(),
    }


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


@app.post("/planificar")
def planificar(p: PeticionPlanificar):
    """Reparte las tiendas entre la flota y ordena cada ruta, todo a la vez.

    Devuelve lo que devolvían `/agrupar` y `/rutear` juntos, con las mismas
    formas, para que la web pueda guardar el resultado sin distinguir de dónde
    vino. Lo nuevo es `motivos`: por qué quedó fuera cada tienda sin asignar."""
    if not p.flota:
        raise HTTPException(400, "Define al menos un tipo de vehículo.")
    df = _a_dataframe(p.tiendas).reset_index(drop=True)
    flota = _expandir_flota(p.flota)
    if len(flota) > 300:
        raise HTTPException(400, "Máximo 300 vehículos por planificación.")

    salida_s = hora_a_segundos(p.hora_salida) or 8 * 3600
    jornada_s = int(p.jornada_h * 3600)
    bultos = df["cantidad_bultos"].values.astype(float)
    servicio = p.servicio_min * 60 + bultos * p.servicio_min_bulto * 60
    ventanas = [
        (int(a), int(b)) if pd.notna(a) and pd.notna(b) and b > a else None
        for a, b in zip(df["ventana_ini_s"], df["ventana_fin_s"])
    ]

    # Las etiquetas van en el mismo orden que las tiendas y que los vehículos
    # de la flota ya desplegada.
    pide = [_etiquetas(t.etiquetas) for t in p.tiendas]
    tiene = []
    for tipo in p.flota:
        tiene.extend([_etiquetas(tipo.etiquetas)] * tipo.cantidad)

    # De qué base sale cada vehículo de la flota ya desplegada.
    bases = [(b.lat, b.lon) for b in p.bases] or None
    base_vehiculo = None
    if p.bases:
        if len(p.bases) > 20:
            raise HTTPException(400, "Máximo 20 bases por planificación.")
        pos_base = {b.id: i for i, b in enumerate(p.bases)}
        base_vehiculo = []
        for tipo in p.flota:
            if tipo.base is not None and tipo.base not in pos_base:
                raise HTTPException(400, f"Un vehículo sale de una base que no se mandó: {tipo.base}")
            base_vehiculo.extend([pos_base.get(tipo.base, 0)] * tipo.cantidad)

    try:
        res = planificar_flota(
            df[["latitud", "longitud"]].values.astype(float),
            bultos, servicio, ventanas, df["prioridad"].values.astype(int), flota,
            p.cd_lat, p.cd_lon,
            motor=p.motor, cerrado=p.cerrado, salida_s=salida_s, jornada_s=jornada_s,
            tiempo_seg=p.tiempo_s, costo_fijo=p.costo_fijo, costo_km=p.costo_km,
            etiquetas_tienda=pide, etiquetas_vehiculo=tiene,
            peso_prioridad=p.peso_prioridad, balanceo=p.balanceo,
            bases=bases, base_vehiculo=base_vehiculo,
        )
    except Exception as e:
        raise HTTPException(500, f"No se pudo planificar: {type(e).__name__}: {e}")

    por_calles = res["matriz"] == "calles"
    aviso_matriz = None
    if p.motor == "osrm" and not por_calles:
        aviso_matriz = (
            f"Con {len(df)} puntos no se puede pedir la matriz por calles al servidor "
            "público: el reparto se decidió con distancias estimadas (línea recta por "
            "1,4). Los kilómetros y el trazado de cada ruta sí son por calles."
        )

    grupos, rutas = [], []
    for i, r in enumerate(res["rutas"]):
        sub = df.iloc[r["orden"]]
        paradas = [{
            "id": str(sub.iloc[n]["id"]),
            "orden": n + 1,
            "codigo": _texto(sub.iloc[n]["codigo_sucursal"]) or "",
            "nombre": _texto(sub.iloc[n]["name_sucursal"]) or "",
            "distrito": _texto(sub.iloc[n]["distrito"]),
            "bultos": int(sub.iloc[n]["cantidad_bultos"]),
            "prioridad": int(sub.iloc[n]["prioridad"]),
            "lat": float(sub.iloc[n]["latitud"]),
            "lon": float(sub.iloc[n]["longitud"]),
            "eta": segundos_a_hora(r["llegadas"].get(t)),
        } for n, t in enumerate(r["orden"])]

        # De dónde sale: su base, o el centro común si no hay varias.
        base = p.bases[r["base_idx"]] if p.bases else None
        origen = (base.lat, base.lon) if base else (p.cd_lat, p.cd_lon)
        coords_orden = ([origen]
                        + [(x["lat"], x["lon"]) for x in paradas]
                        + ([origen] if p.cerrado else []))

        # El trazado y los kilómetros, por calles reales si se pidió y se puede.
        km = r["dist_m"] / 1000.0
        geometria = coords_orden
        motor_txt = "Planificador · " + ("OSRM (calles reales)" if por_calles else "línea recta ×1,4")
        if p.motor == "osrm":
            try:
                puntos, dist_real, _ = osrm_ruta_orden(
                    tuple((lon, lat) for lat, lon in coords_orden))
                geometria, km = puntos, dist_real / 1000.0
                motor_txt = "Planificador · OSRM (calles reales)"
            except Exception:
                pass  # sin trazado: se dibuja en línea recta y se sigue

        cap, mt = r["vehiculo"]
        rutas.append({
            "indice": i,
            "paradas": paradas,
            "orden": [x["id"] for x in paradas],
            "km": round(km, 2),
            "duracion_min": round((r["fin_s"] - salida_s) / 60.0, 1),
            "bultos": int(sub["cantidad_bultos"].sum()),
            "costo": round(p.costo_fijo + km * p.costo_km, 2),
            "salida": segundos_a_hora(salida_s),
            "fin": segundos_a_hora(r["fin_s"]),
            "motor": motor_txt,
            "aviso": aviso_matriz,
            "geometria": [[float(la), float(lo)] for la, lo in geometria],
            "links_maps": links_google_maps(coords_orden),
            "base": base.model_dump() if base else None,
        })
        grupos.append({
            "indice": i,
            "base": base.id if base else None,
            "tiendas": [x["id"] for x in paradas],
            "n_tiendas": len(paradas),
            "bultos": int(sub["cantidad_bultos"].sum()),
            "prioritarias": int((sub["prioridad"] > 0).sum()),
            "capacidad_vehiculo": int(cap),
            "max_tiendas_vehiculo": int(mt),
            "etiquetas_vehiculo": sorted(tiene[r["vehiculo_idx"]]),
            "vuelta": 1,
            "centro": {
                "lat": float(sub["latitud"].mean()),
                "lon": float(sub["longitud"].mean()),
            },
        })

    motivos = {str(df.iloc[t]["id"]): m for t, m in res["sin_asignar"].items()}
    return _sin_nan({
        "grupos": grupos,
        "rutas": rutas,
        "sin_asignar": list(motivos.keys()),
        "motivos": motivos,
        "totales": {
            "rutas": len(rutas),
            "km": round(sum(r["km"] for r in rutas), 2),
            "duracion_min": round(sum(r["duracion_min"] for r in rutas), 1),
            "costo": round(sum(r["costo"] for r in rutas), 2),
            "paradas": sum(len(r["paradas"]) for r in rutas),
        },
        "resumen": {
            "tiendas": int(len(df)),
            "vehiculos_disponibles": len(flota),
            "vehiculos_usados": len(rutas),
            "sin_asignar": len(motivos),
            "matriz": res["matriz"],
            "desde_agrupacion": bool(res["desde_agrupacion"]),
            "balanceo": p.balanceo,
            "peso_prioridad": p.peso_prioridad,
            "con_etiquetas": sum(1 for e in pide if e),
            "bases": len(p.bases),
            "bases_usadas": len({r["base_idx"] for r in res["rutas"]}) if p.bases else 0,
        },
    })


@app.post("/insertar")
def insertar(p: PeticionInsertar):
    """Busca el mejor hueco para cada pedido nuevo en las rutas en marcha.

    No reparte el día otra vez: cada vehículo parte de donde está, lo ya
    gestionado no cuenta y el orden de lo pendiente se respeta. Devuelve, por
    cada ruta que recibe algo, su nuevo orden completo de pendientes."""
    if len(p.nuevas) > 200:
        raise HTTPException(400, "Máximo 200 pedidos por inserción.")
    if len(p.rutas) > 300:
        raise HTTPException(400, "Máximo 300 rutas por inserción.")
    ahora_s = hora_a_segundos(p.hora_actual)
    if ahora_s is None:
        raise HTTPException(400, "La hora actual no se entiende: se espera HH:MM.")

    def parada(t: Tienda) -> dict:
        ini, fin = hora_a_segundos(t.ventana_ini), hora_a_segundos(t.ventana_fin)
        bultos = max(0, int(t.bultos))
        return {
            "id": t.id, "lat": t.lat, "lon": t.lon, "bultos": bultos,
            "servicio_s": p.servicio_min * 60 + bultos * p.servicio_min_bulto * 60,
            "ventana": (ini, fin) if ini is not None and fin is not None and fin > ini else None,
            "prioridad": t.prioridad,
            "etiquetas": _etiquetas(t.etiquetas),
        }

    rutas = [{
        "id": r.id, "lat": r.lat, "lon": r.lon,
        "base": (r.base_lat, r.base_lon) if r.base_lat is not None and r.base_lon is not None else None,
        "hora_s": hora_a_segundos(r.hora) if hora_a_segundos(r.hora) is not None else ahora_s,
        "pendientes": [parada(t) for t in r.pendientes],
        "fijas": r.fijas,
        "cap_libre": r.capacidad_libre,
        "paradas_libres": r.paradas_libres,
        "etiquetas": None if r.etiquetas is None else _etiquetas(r.etiquetas),
    } for r in p.rutas]
    nuevas = [parada(t) for t in p.nuevas]
    if len({t["id"] for t in nuevas}) != len(nuevas):
        raise HTTPException(400, "Hay pedidos repetidos en la petición.")

    try:
        res = insertar_en_rutas(
            rutas, nuevas, p.cd_lat, p.cd_lon,
            motor=p.motor, cerrado=p.cerrado,
            fin_jornada_s=hora_a_segundos(p.fin_jornada),
            reordenar=p.reordenar, peso_prioridad=p.peso_prioridad,
        )
    except Exception as e:
        raise HTTPException(500, f"No se pudo insertar: {type(e).__name__}: {e}")

    return _sin_nan({
        "rutas": [{
            "id": r["id"],
            "orden": r["orden"],
            "etas": {k: segundos_a_hora(v) for k, v in r["llegadas"].items()},
            "fin": segundos_a_hora(r["fin_s"]),
            "fin_antes": segundos_a_hora(r["fin_antes_s"]),
            "min_extra": round((r["fin_s"] - r["fin_antes_s"]) / 60.0, 1),
            "km_extra": round(r["dist_extra_m"] / 1000.0, 2),
            "nuevas": r["nuevas"],
            "reordenada": r["reordenada"],
        } for r in res["rutas"]],
        "inserciones": {k: {
            "ruta": v["ruta"],
            "posicion": v["posicion"],
            "despues_de": v["despues_de"],
            "eta": segundos_a_hora(v["llegada_s"]),
            "km_extra": round(v["dist_extra_m"] / 1000.0, 2),
            "sin_comprobar": v["sin_comprobar"],
        } for k, v in res["inserciones"].items()},
        "sin_asignar": list(res["sin_asignar"].keys()),
        "motivos": res["sin_asignar"],
        "resumen": {
            "pedidos": len(nuevas),
            "colocados": len(res["inserciones"]),
            "sin_asignar": len(res["sin_asignar"]),
            "rutas_tocadas": len(res["rutas"]),
            "matriz": res["matriz"],
        },
    })


@app.post("/reordenar")
def reordenar(p: PeticionReordenar):
    """Ordena lo que le queda a una ruta, o evalúa el orden que se le da.

    Devuelve el orden de antes y el propuesto con sus horas, kilómetros y las
    paradas que llegarían fuera de su ventana. Si el orden que había ya era el
    mejor, `cambio` es falso y los dos son el mismo."""
    if not p.paradas:
        raise HTTPException(400, "La ruta no tiene paradas pendientes.")
    if len(p.paradas) > 300:
        raise HTTPException(400, "Máximo 300 paradas por ruta.")
    hora_s = hora_a_segundos(p.hora)
    if hora_s is None:
        raise HTTPException(400, "La hora no se entiende: se espera HH:MM.")

    paradas = []
    for t in p.paradas:
        ini, fin = hora_a_segundos(t.ventana_ini), hora_a_segundos(t.ventana_fin)
        bultos = max(0, int(t.bultos))
        paradas.append({
            "id": t.id, "lat": t.lat, "lon": t.lon,
            "servicio_s": p.servicio_min * 60 + bultos * p.servicio_min_bulto * 60,
            "ventana": (ini, fin) if ini is not None and fin is not None and fin > ini else None,
        })
    if len({x["id"] for x in paradas}) != len(paradas):
        raise HTTPException(400, "Hay paradas repetidas en la petición.")

    centro = ((p.centro_lat, p.centro_lon)
              if p.centro_lat is not None and p.centro_lon is not None else None)
    try:
        res = reordenar_ruta(
            paradas, p.lat, p.lon, hora_s,
            centro=centro, final=p.final, final_id=p.final_id, fijas=p.fijas,
            ancladas=p.ancladas,
            motor=p.motor, optimizar=p.optimizar, tiempo_seg=p.tiempo_s,
        )
    except Exception as e:
        raise HTTPException(500, f"No se pudo reordenar: {type(e).__name__}: {e}")

    def forma(r: dict) -> dict:
        return {
            "orden": r["orden"],
            "etas": {k: segundos_a_hora(v) for k, v in r["llegadas"].items()},
            "fin": segundos_a_hora(r["fin_s"]),
            "min": round((r["fin_s"] - hora_s) / 60.0, 1),
            "km": round(r["dist_m"] / 1000.0, 2),
            "tarde": r["tarde"],
        }

    return _sin_nan({
        "antes": forma(res["antes"]),
        "despues": forma(res["despues"]),
        "cambio": bool(res["cambio"]),
        "matriz": res["matriz"],
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
