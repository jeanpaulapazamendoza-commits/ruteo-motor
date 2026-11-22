"""Meter pedidos nuevos en rutas que ya están en marcha.

Como `planificar.py` y `calles.py`, este módulo NO viene de la app de
Streamlit: se escribió aquí y se edita aquí.

Qué problema resuelve
---------------------
`/planificar` reparte un día entero desde cero. A media mañana eso ya no
sirve: los vehículos salieron, parte de las paradas están entregadas y el
conductor lleva un orden en la cabeza. Llega un pedido urgente y hay que
decidir a qué ruta va y entre qué dos paradas, sin deshacer nada de lo hecho.

Qué se respeta
--------------
  * Lo ya gestionado no existe para este cálculo: solo cuentan las paradas
    pendientes, y cada vehículo parte de donde está, a la hora que es.
  * El orden de las pendientes no se toca. La nueva entra en el hueco que
    menos rodeo cuesta. Solo si se pide (`reordenar`) se retoca el orden de
    las rutas que recibieron algo, y nunca el de las demás.
  * Las primeras `fijas` pendientes de cada ruta no se mueven ni admiten nada
    delante: el conductor ya va hacia ellas.
  * La carga que le queda libre al vehículo, su máximo de paradas y lo que el
    pedido exige de él (etiquetas).
  * Las horas. Una inserción no puede sacar a ninguna parada de su ventana
    horaria ni pasar la ruta de la jornada. Si una parada ya iba tarde antes,
    no se le exige lo imposible: se exige no empeorarla.

Lo que no encuentra hueco no se fuerza: vuelve con su motivo.

Por qué no se usa el solver
---------------------------
Son pocos pedidos contra rutas ya decididas: probar todos los huecos es
exacto, tarda milisegundos y deja explicar cada decisión («entre tal parada y
tal otra, 2,3 km más»). El solver brilla cuando hay que decidirlo todo.
"""

import math

import numpy as np

from .planificar import (
    PRIORIDAD_M_POR_S,
    PRIORIDAD_NIVEL_MAX,
    _matrices,
)

MOTIVO_SIN_RUTAS = "No hay ninguna ruta en marcha donde meterla."
MOTIVO_ETIQUETA = "Pide {lista} y ninguna ruta lleva un vehículo que lo tenga."
MOTIVO_CARGA = "A ningún vehículo le queda sitio para sus {n} bultos."
MOTIVO_PARADAS = "Todas las rutas están ya en su máximo de paradas."
MOTIVO_VENTANA = "Su ventana horaria cierra antes de que ningún vehículo pueda llegar."
MOTIVO_TIEMPO = ("Ninguna ruta la admite sin pasarse de la jornada o sin sacar "
                 "otra parada de su ventana horaria.")

# Cuántas rutas se miran por pedido en la primera pasada: las más cercanas en
# línea recta. Mirarlas todas obligaría a pedir la matriz de calles del
# despacho entero para decidir dónde va un solo punto.
RUTAS_VECINAS = 8

# Una mejora de menos de esto al reordenar no compensa cambiarle el orden al
# conductor.
MEJORA_MINIMA_M = 50.0


class _Ruta:
    def __init__(self, pos, datos, nodo_inicio, nodos, nodo_base, cerrado):
        self.pos = pos                      # posición en la lista de entrada
        self.id = datos["id"]
        self.inicio = nodo_inicio
        self.base = nodo_base               # su centro: de ahí salió
        self.fin = nodo_base if cerrado else None   # y ahí vuelve, si es cerrada
        self.hora = float(datos["hora_s"])
        self.seq = list(nodos)              # nodos de las paradas pendientes
        self.fijas = max(0, min(int(datos.get("fijas") or 0), len(self.seq)))
        self.cap_libre = datos.get("cap_libre")
        self.paradas_libres = datos.get("paradas_libres")
        etiquetas = datos.get("etiquetas")
        self.etiquetas = None if etiquetas is None else set(etiquetas)
        self.orden_antes = list(nodos)
        self.nuevas = []
        self.antes = None                   # (llegadas, fin, dist) antes de tocarla
        self.final = None                   # lo mismo, ya con lo insertado
        self.reordenada = False


class _Mapa:
    """Distancias y tiempos entre los nodos de una pasada."""

    def __init__(self, coords, nodos, motor):
        self.loc = {g: i for i, g in enumerate(nodos)}
        dist, dur, self.matriz = _matrices(coords[nodos], motor)
        # Listas de Python: el cálculo lee celdas sueltas millones de veces,
        # y ahí una lista es varias veces más rápida que un array.
        self.D = np.asarray(dist, dtype=float).tolist()
        self.T = np.asarray(dur, dtype=float).tolist()


def _recorrer(ruta, seq, mapa, serv, vent):
    """Horas de llegada a cada parada de `seq`, hora de fin y metros."""
    D, T, loc = mapa.D, mapa.T, mapa.loc
    fin_nodo = ruta.fin
    t = ruta.hora
    prev = loc[ruta.inicio]
    metros = 0.0
    llegadas = []
    for n in seq:
        i = loc[n]
        t += T[prev][i]
        metros += D[prev][i]
        v = vent[n]
        if v is not None and t < v[0]:
            t = v[0]                        # llegó antes de que abran: espera
        llegadas.append(t)
        t += serv[n]
        prev = i
    if fin_nodo is not None:
        i = loc[fin_nodo]
        t += T[prev][i]
        metros += D[prev][i]
    return llegadas, t, metros


def _coste(seq, llegadas, metros, peso):
    """Lo que se minimiza: metros, más la espera de las prioritarias."""
    c = metros
    for n, t in zip(seq, llegadas):
        if peso[n]:
            c += peso[n] * t
    return c


def _admisible(seq, llegadas, fin, antes, fin_antes, vent, fin_jornada):
    """¿Deja a cada parada tan bien como estaba, o dentro de su ventana?

    `antes` es la hora a la que llegaba cada parada que ya estaba en la ruta.
    Una que ya iba tarde no puede retrasarse más; una nueva tiene que entrar
    en su ventana. Lo mismo con la jornada: si la ruta ya se pasaba, no puede
    pasarse más."""
    for n, t in zip(seq, llegadas):
        v = vent[n]
        if v is None:
            continue
        tope = max(v[1], antes.get(n, -math.inf))
        if t > tope + 1e-6:
            return False
    if fin_jornada is not None and fin > max(fin_jornada, fin_antes) + 1e-6:
        return False
    return True


def _mejor_hueco(ruta, p, mapa, serv, vent, peso, bultos, pide, fin_jornada):
    """El hueco más barato de `ruta` para el pedido `p`.

    Devuelve (coste_extra, posición) o (None, por_qué_no)."""
    if pide[p] and ruta.etiquetas is not None and not pide[p] <= ruta.etiquetas:
        return None, "etiqueta"
    if ruta.cap_libre is not None and bultos[p] > ruta.cap_libre:
        return None, "carga"
    if ruta.paradas_libres is not None and ruta.paradas_libres < 1:
        return None, "paradas"

    lleg0, fin0, met0 = _recorrer(ruta, ruta.seq, mapa, serv, vent)
    antes = dict(zip(ruta.seq, lleg0))
    coste0 = _coste(ruta.seq, lleg0, met0, peso)

    mejor = None
    llega_a_tiempo = False
    for pos in range(ruta.fijas, len(ruta.seq) + 1):
        seq = ruta.seq[:pos] + [p] + ruta.seq[pos:]
        lleg, fin, met = _recorrer(ruta, seq, mapa, serv, vent)
        v = vent[p]
        if v is None or lleg[pos] <= v[1] + 1e-6:
            llega_a_tiempo = True
        if not _admisible(seq, lleg, fin, antes, fin0, vent, fin_jornada):
            continue
        extra = _coste(seq, lleg, met, peso) - coste0
        if mejor is None or extra < mejor[0]:
            mejor = (extra, pos)
    if mejor is None:
        return None, "tiempo" if llega_a_tiempo else "ventana"
    return mejor


def _reordenar(ruta, mapa, serv, vent, peso, fin_jornada):
    """Retoca el orden de las pendientes que se pueden mover.

    Dos movimientos, repetidos hasta que ninguno mejora: sacar una parada y
    meterla en otro sitio, e invertir un tramo. Solo se acepta lo que ahorra
    recorrido sin empeorar ninguna hora comprometida."""
    lleg, fin_tope, met = _recorrer(ruta, ruta.seq, mapa, serv, vent)
    antes = dict(zip(ruta.seq, lleg))
    coste = _coste(ruta.seq, lleg, met, peso)
    coste_inicial = coste
    seq_inicial = list(ruta.seq)
    f = ruta.fijas
    mejorado = True
    vueltas = 0
    while mejorado and vueltas < 50:
        mejorado = False
        vueltas += 1
        n = len(ruta.seq)
        candidatas = []
        for i in range(f, n):               # mover una parada
            resto = ruta.seq[:i] + ruta.seq[i + 1:]
            for j in range(f, n):
                if j != i:
                    candidatas.append(resto[:j] + [ruta.seq[i]] + resto[j:])
        for i in range(f, n - 1):           # invertir un tramo
            for j in range(i + 1, n):
                candidatas.append(ruta.seq[:i] + ruta.seq[i:j + 1][::-1] + ruta.seq[j + 1:])
        for seq in candidatas:
            lleg, fin, met = _recorrer(ruta, seq, mapa, serv, vent)
            c = _coste(seq, lleg, met, peso)
            if c < coste - 1.0 and _admisible(seq, lleg, fin, antes, fin_tope, vent, fin_jornada):
                ruta.seq, coste, mejorado = seq, c, True
                break
    if coste_inicial - coste < MEJORA_MINIMA_M:
        # No compensa: se deja el orden que tenía tras insertar.
        ruta.seq = seq_inicial
        return False
    return True


def insertar_en_rutas(rutas, nuevas, cd_lat, cd_lon, *, motor="osrm", cerrado=True,
                      fin_jornada_s=None, reordenar=False, peso_prioridad=1.0):
    """Busca el mejor hueco para cada pedido nuevo en las rutas en marcha.

    rutas: lista de dicts con
        id          identificador de la ruta
        lat, lon    dónde está el vehículo (None = en su centro)
        base        `(lat, lon)` de su centro, si no es el común: de ahí sale
                    y ahí vuelve
        hora_s      desde qué hora está libre en ese punto
        pendientes  paradas que le quedan, en su orden: dicts con id, lat,
                    lon, bultos, servicio_s, ventana `(ini_s, fin_s)` o None,
                    prioridad
        fijas       cuántas de las primeras pendientes no se tocan
        cap_libre   bultos que aún admite (None = sin límite)
        paradas_libres  paradas que aún admite (None = sin límite)
        etiquetas   lo que tiene el vehículo (None = no se sabe: no restringe)
    nuevas: lista de dicts como los de `pendientes`, más `etiquetas`.

    Devuelve un dict con
        rutas        solo las que recibieron algo: {id, orden, llegadas,
                     fin_s, fin_antes_s, dist_extra_m, nuevas, reordenada}
        inserciones  {id del pedido: {ruta, posicion, despues_de, llegada_s,
                     dist_extra_m, sin_comprobar}}
        sin_asignar  {id del pedido: motivo}
        matriz       'calles' o 'recta'
    """
    if not nuevas:
        return {"rutas": [], "inserciones": {}, "sin_asignar": {}, "matriz": "calles"}
    if not rutas:
        return {"rutas": [], "inserciones": {},
                "sin_asignar": {p["id"]: MOTIVO_SIN_RUTAS for p in nuevas},
                "matriz": "calles"}

    # ── Un número de nodo para cada punto: 0 el centro, luego todo lo demás ──
    coords = [(float(cd_lat), float(cd_lon))]
    ident, serv, vent, peso, bultos, pide = [None], [0.0], [None], [0.0], [0], [set()]

    def _nodo(lat, lon, d=None):
        coords.append((float(lat), float(lon)))
        d = d or {}
        ident.append(d.get("id"))
        serv.append(float(d.get("servicio_s") or 0.0))
        vent.append(d.get("ventana"))
        prio = int(d.get("prioridad") or 0)
        peso.append(PRIORIDAD_M_POR_S * peso_prioridad * min(prio, PRIORIDAD_NIVEL_MAX)
                    if prio > 0 else 0.0)
        bultos.append(int(round(float(d.get("bultos") or 0))))
        pide.append(set(d.get("etiquetas") or ()))
        return len(coords) - 1

    estado = []
    nodo_de_base = {}
    for pos, r in enumerate(rutas):
        # El centro de la ruta: el suyo si lo trae, o el común.
        base = 0
        if r.get("base") is not None:
            clave = (round(float(r["base"][0]), 6), round(float(r["base"][1]), 6))
            if clave not in nodo_de_base:
                nodo_de_base[clave] = _nodo(*clave)
            base = nodo_de_base[clave]
        if r.get("lat") is None or r.get("lon") is None:
            inicio = base
        else:
            inicio = _nodo(r["lat"], r["lon"])
        nodos = [_nodo(s["lat"], s["lon"], s) for s in r.get("pendientes", [])]
        estado.append(_Ruta(pos, r, inicio, nodos, base, cerrado))
    pedidos = [_nodo(p["lat"], p["lon"], p) for p in nuevas]
    coords = np.asarray(coords, dtype=float)

    # ── Qué rutas mira cada pedido, de la más cercana a la más lejana ───────
    def _cerca(p, r):
        pts = [r.inicio] + r.seq + ([r.base] if cerrado else [])
        d = coords[pts] - coords[p]
        d[:, 1] *= math.cos(math.radians(coords[p][0]))
        return float(np.min(np.hypot(d[:, 0], d[:, 1])))

    cola = {p: sorted(estado, key=lambda r, p=p: _cerca(p, r)) for p in pedidos}
    porque = {p: set() for p in pedidos}
    colocado = {}                           # pedido -> (ruta, metros extra)
    matrices = set()
    args = (serv, vent, peso, bultos, pide, fin_jornada_s)

    faltan = list(pedidos)
    while faltan:
        # Las rutas de esta pasada: las vecinas que cada pedido aún no probó.
        tanda, vistos = [], set()
        for p in faltan:
            for r in cola[p][:RUTAS_VECINAS]:
                if r.pos not in vistos:
                    vistos.add(r.pos)
                    tanda.append(r)
            cola[p] = cola[p][RUTAS_VECINAS:]
        if not tanda:
            break
        nodos = sorted({0} | set(faltan)
                       | {n for r in tanda for n in [r.inicio, r.base] + r.seq})
        mapa = _Mapa(coords, nodos, motor)
        matrices.add(mapa.matriz)
        for r in tanda:
            # Una ruta puede volver en otra pasada, ya con algo dentro: su
            # punto de comparación es el de la primera vez.
            if r.antes is None:
                r.antes = _recorrer(r, r.seq, mapa, serv, vent)

        # Primero el pedido que más pierde si no se le da su mejor ruta.
        hueco = {}
        libres = list(faltan)
        while libres:
            opciones = []
            for p in libres:
                validas = []
                for r in tanda:
                    if (p, r.pos) not in hueco:
                        hueco[(p, r.pos)] = _mejor_hueco(r, p, mapa, *args)
                    extra, dato = hueco[(p, r.pos)]
                    if extra is None:
                        porque[p].add(dato)
                    else:
                        validas.append((extra, r.pos, dato, r))
                if not validas:
                    continue
                validas.sort(key=lambda x: (x[0], x[1]))
                pena = validas[1][0] - validas[0][0] if len(validas) > 1 else math.inf
                opciones.append((peso[p] > 0, pena, -validas[0][0], -p, p, validas[0]))
            if not opciones:
                break
            *_, p, (_, _, pos, r) = max(opciones, key=lambda x: x[:4])
            _, _, met0 = _recorrer(r, r.seq, mapa, serv, vent)
            r.seq.insert(pos, p)
            _, _, met1 = _recorrer(r, r.seq, mapa, serv, vent)
            r.nuevas.append(p)
            if r.cap_libre is not None:
                r.cap_libre -= bultos[p]
            if r.paradas_libres is not None:
                r.paradas_libres -= 1
            colocado[p] = (r, met1 - met0)
            libres.remove(p)
            for q in libres:
                hueco.pop((q, r.pos), None)

        # Cerrar las rutas que recibieron algo, con la matriz de esta pasada.
        for r in tanda:
            if not r.nuevas:
                continue
            r.reordenada = bool(reordenar) and _reordenar(
                r, mapa, serv, vent, peso, fin_jornada_s)
            r.final = _recorrer(r, r.seq, mapa, serv, vent)
        faltan = [p for p in faltan if p not in colocado]

    # ── Respuesta ──────────────────────────────────────────────────────────
    salida_rutas = []
    inserciones = {}
    for r in estado:
        if not r.nuevas:
            continue
        llegadas, fin, metros = r.final
        _, fin_antes, metros_antes = r.antes
        salida_rutas.append({
            "id": r.id,
            "orden": [ident[n] for n in r.seq],
            "llegadas": {ident[n]: int(round(t)) for n, t in zip(r.seq, llegadas)},
            "fin_s": int(round(fin)),
            "fin_antes_s": int(round(fin_antes)),
            "dist_extra_m": float(metros - metros_antes),
            "nuevas": [ident[n] for n in r.nuevas],
            "reordenada": r.reordenada,
        })
        for n in r.nuevas:
            i = r.seq.index(n)
            inserciones[ident[n]] = {
                "ruta": r.id,
                "posicion": i,
                "despues_de": ident[r.seq[i - 1]] if i > 0 else None,
                "llegada_s": int(round(llegadas[i])),
                "dist_extra_m": float(colocado[n][1]),
                # Pide algo del vehículo y de esta ruta no se sabe qué lleva.
                "sin_comprobar": sorted(pide[n]) if pide[n] and r.etiquetas is None else [],
            }

    sin_asignar = {}
    for p in pedidos:
        if p in colocado:
            continue
        q = porque[p]
        if "tiempo" in q:
            motivo = MOTIVO_TIEMPO
        elif "ventana" in q:
            motivo = MOTIVO_VENTANA
        elif "carga" in q:
            motivo = MOTIVO_CARGA.format(n=bultos[p])
        elif "paradas" in q:
            motivo = MOTIVO_PARADAS
        elif "etiqueta" in q:
            motivo = MOTIVO_ETIQUETA.format(
                lista=", ".join(f"«{e}»" for e in sorted(pide[p])))
        else:
            motivo = MOTIVO_SIN_RUTAS
        sin_asignar[ident[p]] = motivo

    return {
        "rutas": salida_rutas,
        "inserciones": inserciones,
        "sin_asignar": sin_asignar,
        "matriz": "recta" if "recta" in matrices or not matrices else "calles",
    }
