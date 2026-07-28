"""Extraído de app.py (Streamlit) — no editar a mano sin re-extraer."""


import math
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances



def kmeans_balanceado(Xm, k, capacidad, pesos=None, seed=42, n_iter=15):
    """K-Means con restricción de capacidad. `capacidad` puede ser un máximo de
    TIENDAS (pesos = None -> cada tienda pesa 1) o un máximo de BULTOS
    (pesos = bultos de cada tienda). Heurística tipo 'balanced k-means':
    en cada iteración los puntos se asignan por orden de urgencia al cluster
    más cercano que aún tenga cupo. Escala bien a miles de puntos.
    Devuelve (labels, hubo_sobrecupo)."""
    n = len(Xm)
    if pesos is None:
        pesos = np.ones(n)
    pesos = np.asarray(pesos, dtype=float)
    base = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xm)
    centros = base.cluster_centers_.copy()
    labels = np.full(n, -1)
    sobrecupo = False
    for _ in range(n_iter):
        D = pairwise_distances(Xm, centros)
        if k >= 2:
            mejores2 = np.partition(D, 1, axis=1)
            orden_puntos = np.argsort(mejores2[:, 0] - mejores2[:, 1])
        else:
            orden_puntos = np.arange(n)
        nuevo = np.full(n, -1)
        carga = np.zeros(k, dtype=float)
        sobrecupo = False
        for p in orden_puntos:
            colocado = False
            for c in np.argsort(D[p]):
                if carga[c] + pesos[p] <= capacidad:
                    nuevo[p] = c
                    carga[c] += pesos[p]
                    colocado = True
                    break
            if not colocado:
                # ningún cluster tiene cupo exacto: usar el de mayor cupo libre
                c = int(np.argmin(carga + pesos[p]))
                nuevo[p] = c
                carga[c] += pesos[p]
                sobrecupo = True
        if np.array_equal(nuevo, labels):
            break
        labels = nuevo
        for c in range(k):
            mask = labels == c
            if mask.any():
                centros[c] = Xm[mask].mean(axis=0)
    # Pulido: corrige asignaciones lejanas mediante movimientos e intercambios
    labels = _refinar_intercambios(Xm, labels, np.full(k, float(capacidad)),
                                   np.full(k, n + 1), pesos)
    return labels, sobrecupo


def _refinar_intercambios(Xm_sub, labels, caps, maxt, pesos_sub, n_pasadas=4):
    """Pulido local de la asignación: mueve o INTERCAMBIA tiendas entre grupos
    cuando eso reduce la distancia total sin violar capacidades. Corrige las
    asignaciones 'lejanas' que deja el empaque cuando los grupos van muy llenos."""
    labels = labels.copy()
    k = len(caps)
    for _ in range(n_pasadas):
        centros = np.vstack([
            Xm_sub[labels == c].mean(axis=0) if (labels == c).any()
            else np.full(Xm_sub.shape[1], 1e12) for c in range(k)
        ])
        D = pairwise_distances(Xm_sub, centros)
        carga = np.array([pesos_sub[labels == c].sum() for c in range(k)])
        cuenta = np.array([(labels == c).sum() for c in range(k)])
        asignados = np.where(labels >= 0)[0]
        if len(asignados) == 0:
            break
        regret = D[asignados, labels[asignados]] - D[asignados].min(axis=1)
        orden = asignados[np.argsort(-regret)]
        mejorado = False
        for p in orden:
            c_act = labels[p]
            for c_new in np.argsort(D[p])[:6]:
                if c_new == c_act:
                    break  # su grupo ya es el más cercano disponible
                ganancia_mov = D[p, c_act] - D[p, c_new]
                if ganancia_mov <= 1e-9:
                    break
                # 1) mover directo si el grupo cercano tiene cupo
                if carga[c_new] + pesos_sub[p] <= caps[c_new] and cuenta[c_new] < maxt[c_new]:
                    labels[p] = c_new
                    carga[c_act] -= pesos_sub[p]; carga[c_new] += pesos_sub[p]
                    cuenta[c_act] -= 1; cuenta[c_new] += 1
                    mejorado = True
                    break
                # 2) intercambiar con una tienda de ese grupo que prefiera el mío
                idx_new = np.where(labels == c_new)[0]
                if len(idx_new) == 0:
                    continue
                delta_q = D[idx_new, c_act] - D[idx_new, c_new]
                viable = ((carga[c_new] - pesos_sub[idx_new] + pesos_sub[p] <= caps[c_new])
                          & (carga[c_act] - pesos_sub[p] + pesos_sub[idx_new] <= caps[c_act])
                          & (delta_q < ganancia_mov - 1e-9))
                cand = np.where(viable)[0]
                if len(cand):
                    q = idx_new[cand[np.argmin(delta_q[cand])]]
                    labels[p], labels[q] = c_new, c_act
                    carga[c_act] += pesos_sub[q] - pesos_sub[p]
                    carga[c_new] += pesos_sub[p] - pesos_sub[q]
                    mejorado = True
                    break
        if not mejorado:
            break
    return labels


def _asignar_vehiculos(Xm_sub, pesos_sub, prio_sub, vehiculos, seed=42, n_iter=12):
    """Balanced k-means con capacidades HETEROGÉNEAS (una por vehículo).
    vehiculos: lista de (capacidad_bultos, max_tiendas); max_tiendas=0 -> sin límite.
    Los vehículos más grandes se asignan a las zonas con más demanda.
    Las tiendas PRIORITARIAS se colocan primero (si falta cupo, quedan fuera
    las normales). Devuelve labels locales; -1 = sin cupo en esta pasada."""
    n = len(Xm_sub)
    k = len(vehiculos)
    base = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xm_sub)
    centros = base.cluster_centers_.copy()
    demanda_ini = np.array([pesos_sub[base.labels_ == c].sum() for c in range(k)])
    veh_orden = sorted(vehiculos, key=lambda t: -t[0])
    caps = np.zeros(k)
    maxt = np.zeros(k, dtype=int)
    veh_por_cluster = [None] * k  # qué vehículo real atiende cada cluster
    for rank, c in enumerate(np.argsort(-demanda_ini)):
        caps[c] = veh_orden[rank][0]
        maxt[c] = veh_orden[rank][1] if veh_orden[rank][1] > 0 else n
        veh_por_cluster[c] = veh_orden[rank]
    nivel = np.where(prio_sub > 0, prio_sub, 99)  # prioridad 1 va primero; 0 = normal al final
    labels = np.full(n, -1)
    for _ in range(n_iter):
        D = pairwise_distances(Xm_sub, centros)
        if k >= 2:
            mejores2 = np.partition(D, 1, axis=1)
            urgencia = mejores2[:, 0] - mejores2[:, 1]
        else:
            urgencia = D[:, 0]
        orden_puntos = np.lexsort((urgencia, nivel))
        nuevo = np.full(n, -1)
        carga = np.zeros(k)
        cuenta = np.zeros(k, dtype=int)
        for p in orden_puntos:
            for c in np.argsort(D[p]):
                if carga[c] + pesos_sub[p] <= caps[c] and cuenta[c] < maxt[c]:
                    nuevo[p] = c
                    carga[c] += pesos_sub[p]
                    cuenta[c] += 1
                    break
        if np.array_equal(nuevo, labels):
            break
        labels = nuevo
        for c in range(k):
            mask = labels == c
            if mask.any():
                centros[c] = Xm_sub[mask].mean(axis=0)

    # Pulido: corrige asignaciones lejanas mediante movimientos e intercambios
    labels = _refinar_intercambios(Xm_sub, labels, caps, maxt, pesos_sub)

    # Intentar colocar pendientes en el cupo que liberó el pulido (prioritarias primero)
    pendientes = np.where(labels < 0)[0]
    if len(pendientes):
        centros = np.vstack([
            Xm_sub[labels == c].mean(axis=0) if (labels == c).any()
            else np.full(Xm_sub.shape[1], 1e12) for c in range(k)
        ])
        D = pairwise_distances(Xm_sub, centros)
        carga = np.array([pesos_sub[labels == c].sum() for c in range(k)])
        cuenta = np.array([(labels == c).sum() for c in range(k)])
        for p in pendientes[np.argsort(nivel[pendientes])]:
            for c in np.argsort(D[p]):
                if carga[c] + pesos_sub[p] <= caps[c] and cuenta[c] < maxt[c]:
                    labels[p] = c
                    carga[c] += pesos_sub[p]
                    cuenta[c] += 1
                    break
    return labels, veh_por_cluster


def _fusionar_clusters(Xm_all, labels, info, pesos_all):
    """Consolidación de rutas: si dos grupos VECINOS caben completos en uno de
    sus dos vehículos (bultos y máx. tiendas), se fusionan en una sola ruta.
    Elimina rutas fragmentadas con vehículos medio vacíos en zonas dispersas
    y libera vehículos. Se repite hasta que ninguna fusión sea factible."""
    labels = labels.copy()
    info = {r: dict(v) for r, v in info.items()}
    while len(info) >= 2:
        ids = sorted(info.keys())
        centros = np.vstack([Xm_all[labels == r].mean(axis=0) for r in ids])
        carga = np.array([pesos_all[labels == r].sum() for r in ids])
        cuenta = np.array([(labels == r).sum() for r in ids])
        Dc = pairwise_distances(centros)
        np.fill_diagonal(Dc, np.inf)
        mejor = None  # (distancia, i, j, índice del vehículo destino)
        for i in range(len(ids)):
            for j in np.argsort(Dc[i])[:2]:  # solo sus 2 vecinos más cercanos
                j = int(j)
                if mejor is not None and Dc[i, j] >= mejor[0]:
                    continue
                # ¿caben juntos? probar primero el vehículo más pequeño de los dos
                for d_idx in sorted((i, j), key=lambda t: info[ids[t]]["capacidad"]):
                    cap_d = info[ids[d_idx]]["capacidad"]
                    mt_d = info[ids[d_idx]]["max_tiendas"] or 10 ** 9
                    if carga[i] + carga[j] <= cap_d and cuenta[i] + cuenta[j] <= mt_d:
                        mejor = (Dc[i, j], i, j, d_idx)
                        break
        if mejor is None:
            break
        _, i, j, d_idx = mejor
        destino = ids[d_idx]
        origen = ids[j] if d_idx == i else ids[i]
        labels[labels == origen] = destino
        del info[origen]
    # reindexar rutas como 0..k-1
    ids = sorted(info.keys())
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(ids)}
    labels_out = labels.copy()
    for viejo, nuevo in mapa.items():
        labels_out[labels == viejo] = nuevo
    return labels_out, {mapa[v]: info[v] for v in ids}


def asignar_flota(Xm_all, pesos_all, prio_all, flota, max_vueltas, usar_holgura=True, seed=42):
    """Asigna tiendas a una flota heterogénea por vueltas.
    flota: tupla de (capacidad_bultos, max_tiendas) por vehículo disponible.
    En cada vuelta se usan los vehículos mínimos necesarios (se agregan más si
    la geografía no permite empacar); lo que no entra pasa a la siguiente vuelta.
    Devuelve (labels, info): labels[i] = id de ruta o -1 (no asignada);
    info[ruta] = {'capacidad', 'max_tiendas', 'vuelta'}."""
    n = len(Xm_all)
    labels_global = np.full(n, -1)
    info = {}
    siguiente = 0
    pendientes = np.arange(n)
    flota_orden = sorted(flota, key=lambda t: -t[0])
    max_cap = flota_orden[0][0]
    for vuelta in range(1, max_vueltas + 1):
        # tiendas imposibles: su pedido excede al vehículo más grande
        posibles = pendientes[pesos_all[pendientes] <= max_cap]
        if len(posibles) == 0:
            break
        demanda = float(pesos_all[posibles].sum())
        usados = []
        cap_acum, cupo_tiendas = 0.0, 0
        for cap, mt in flota_orden:
            if cap_acum >= demanda and cupo_tiendas >= len(posibles):
                break
            usados.append((cap, mt))
            cap_acum += cap
            cupo_tiendas += mt if mt > 0 else len(posibles)
        if usar_holgura:
            # ~30% más de vehículos que el mínimo: grupos al ~75-85% de llenado
            # producen zonas geográficas mucho más compactas y sin cruces
            extra = math.ceil(len(usados) * 0.3)
            for cap, mt in flota_orden[len(usados):len(usados) + extra]:
                usados.append((cap, mt))
        sub_labels = np.full(len(posibles), -1)
        veh_cluster = []
        for _ in range(9):  # si el empaque no cierra, sumar vehículos de la flota
            usados = usados[:max(1, min(len(usados), len(posibles)))]
            sub_labels, veh_cluster = _asignar_vehiculos(
                Xm_all[posibles], pesos_all[posibles],
                prio_all[posibles], usados, seed)
            if (sub_labels < 0).sum() == 0 or len(usados) >= len(flota_orden) \
                    or len(usados) >= len(posibles):
                break
            usados.append(flota_orden[len(usados)])
        for c in range(len(veh_cluster)):
            mask_c = sub_labels == c
            if not mask_c.any():
                continue
            labels_global[posibles[mask_c]] = siguiente
            info[siguiente] = {"capacidad": int(veh_cluster[c][0]),
                               "max_tiendas": int(veh_cluster[c][1]), "vuelta": vuelta}
            siguiente += 1
        pendientes = np.where(labels_global < 0)[0]
        if len(pendientes) == 0:
            break
    # Consolidación final: fusionar grupos vecinos que caben en un solo vehículo
    if len(info) >= 2:
        labels_global, info = _fusionar_clusters(Xm_all, labels_global, info, pesos_all)
    return labels_global, info
