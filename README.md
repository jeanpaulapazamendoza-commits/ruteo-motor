# Motor de ruteo — API

Expone como servicio HTTP el mismo motor de optimización que usa la app de
Streamlit. **No reimplementa nada**: las funciones se extraen de
`ruteo-tiendas-v2/app.py` con `extraer_motor.py`, así que ambas aplicaciones
comparten exactamente el mismo algoritmo.

## Arrancar

```bash
pip install -r requirements.txt
python -m uvicorn api:app --port 8010 --reload
```

Documentación interactiva en <http://localhost:8010/docs>.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/salud` | Comprobación de vida |
| POST | `/agrupar` | Agrupa tiendas por capacidad, bultos, flota o K-Means |
| POST | `/rutear` | Calcula la ruta óptima de cada grupo (OR-Tools) |
| POST | `/planificar` | Reparte y ordena en un solo cálculo, con capacidad, ventanas y jornada como límites |
| POST | `/insertar` | Busca el mejor hueco para pedidos nuevos en rutas que ya están en marcha |
| POST | `/reordenar` | Ordena lo que le queda a una sola ruta, o evalúa el orden que se le da |
| POST | `/tiendas-en-sector` | Qué tiendas caen dentro de un polígono del mapa |

### `/agrupar`

Admite los mismos modos que Streamlit:

- `criterio: "tiendas"` — máximo de paradas por ruta.
- `criterio: "bultos"` — máximo de carga por ruta.
- `criterio: "flota"` — flota heterogénea (`[{cantidad, capacidad, max_tiendas}]`),
  con `vueltas` (1-3) y `uso_flota` (`compacto` = ~30 % más vehículos y zonas
  más limpias, o `minimo` = rutas más llenas).
- `modo: "clasico"` — K-Means con K fijo.
- `grupos_manuales` — respeta una selección hecha a mano en el mapa.

Devuelve grupos con nº de tiendas, bultos, prioritarias, vehículo asignado,
vuelta y centro; más la lista de tiendas `sin_asignar`.

### `/rutear`

Parámetros: `motor` (`osrm` calles reales / `haversine` línea recta / `ors`),
`cerrado`, `hora_salida`, `servicio_min`, `servicio_min_bulto`, `jornada_h`,
`costo_fijo`, `costo_km`, `tiempo_tsp`.

Respeta **ventanas horarias** y **prioridades** de cada tienda, y devuelve por
ruta: paradas en orden con su ETA, km, duración, costo, geometría para el mapa
y links de Google Maps por tramos.

### `/planificar`

Además de los parámetros de `/rutear`:

- `flota[].etiquetas` y `tiendas[].etiquetas` — una tienda solo va en un
  vehículo que tenga todas sus etiquetas.
- `peso_prioridad` (0–5, por defecto 1) — cuánto se adelanta a las tiendas
  prioritarias. 0 = solo son las últimas en quedarse fuera.
- `balanceo` (`ninguno`, `tiempo`, `paradas`) — rutas parejas.
- `bases[]` (`id`, `nombre`, `lat`, `lon`) y `flota[].base` — flota repartida
  entre varios centros. Cada vehículo sale de su base y, si la ruta es
  cerrada, vuelve a ella; un tipo sin base sale de la primera. Sin `bases`,
  todos salen de `cd_lat`/`cd_lon`. Cada ruta devuelve su `base`.

Devuelve `motivos`: por qué quedó fuera cada tienda sin asignar.

### `/insertar`

Para pedidos que llegan con el reparto ya en la calle. No reparte el día otra
vez: recibe cada ruta tal como está ahora y decide a cuál va cada pedido nuevo
y entre qué dos paradas.

- `rutas[]` — por ruta: `lat`/`lon` (dónde está el vehículo; vacío = en el
  centro), `hora` (desde cuándo está libre ahí; vacío = `hora_actual`),
  `pendientes` (lo que le queda, en su orden; lo gestionado no se manda),
  `base_lat`/`base_lon` (su centro, si no es el común),
  `fijas` (cuántas de las primeras no se tocan), `capacidad_libre`,
  `paradas_libres` y `etiquetas` (vacío = no se sabe: no restringe, y la
  respuesta lo avisa en `sin_comprobar`).
- `nuevas[]` — los pedidos, con la misma forma que una tienda.
- `fin_jornada` (HH:MM) — hora a la que tiene que haber terminado todo.
- `reordenar` — permite retocar el orden de las rutas que reciban algo, si
  ahorra recorrido. Las demás nunca se tocan.

Una inserción no puede sacar a ninguna parada de su ventana ni pasar la ruta
de la jornada. Si una parada ya iba tarde, se exige no empeorarla.

Devuelve, por cada ruta que recibe algo, su nuevo `orden` completo de
pendientes con `etas`, `km_extra` y `min_extra`; por cada pedido, su
`inserciones` (ruta, posición, tras qué parada, hora); y `motivos` para lo
que no encontró hueco. No usa el solver: prueba todos los huecos.

### `/reordenar`

Una ruta que ya existe y la pregunta «¿en qué orden conviene hacer lo que
queda?».

- `paradas[]` — las pendientes, en su orden actual.
- `lat`, `lon`, `hora` — de dónde parte el vehículo y a qué hora.
- `final` — `centro` (vuelve a `centro_lat`/`centro_lon`), `parada` (termina
  en `final_id`) o `libre` (termina donde convenga).
- `fijas` — cuántas de las primeras no se mueven.
- `optimizar` — con `false` no cambia nada: solo dice cuánto recorre y a qué
  hora llega el orden dado. Es lo que se usa cuando el orden lo pone una
  persona a mano.

Las ventanas horarias se priorizan pero no bloquean: un minuto de retraso
pesa como 3,6 km de rodeo. Devuelve `antes` y `despues` con `orden`, `etas`,
`km`, `fin` y las paradas que llegarían `tarde`; `cambio` es falso si el
orden que había ya era el mejor.

## Servidor de calles

Las distancias y tiempos por calles salen de OSRM. Sin configurar nada se usa
el servidor público, que alcanza para unas 400 paradas por cálculo: la matriz
se pide por tandas de 10 000 celdas, a una consulta por segundo. Con más
paradas el reparto se decide con distancias estimadas (línea recta × 1,4).

Para pasar de ahí hace falta un servidor propio. Se le indica al motor con
variables de entorno:

| Variable | Qué es | Por defecto |
|---|---|---|
| `OSRM_URL` | Dirección del servidor propio | el público |
| `OSRM_MAX_TABLA` | Lado de la mayor tabla que acepta (`--max-table-size`) | 100 |
| `OSRM_MAX_COORDS` | Puntos que caben en la dirección de una consulta | 350 |
| `OSRM_PAUSA_S` | Segundos entre consultas | 1 público, 0 propio |
| `OSRM_MAX_CONSULTAS` | Consultas aceptadas para una sola matriz | 16 público, 400 propio |

`GET /salud` dice cuál está en uso y hasta cuántos puntos llega.

## Re-extraer el motor

Si cambias el algoritmo en la app de Streamlit:

```bash
cd ..
python extraer_motor.py
```

Reescribe `motor_ruteo/*.py` quitando los decoradores de Streamlit
(`@st.cache_data` → `@lru_cache`). No edites esos archivos a mano.
`planificar.py`, `calles.py`, `insertar.py` y `reordenar.py` no vienen de
Streamlit: son de este servicio y la extracción no los toca.

## Despliegue

Pensado para Render o Railway (Vercel no ejecuta procesos Python persistentes).
Define `ORIGENES_PERMITIDOS` con la URL del frontend, separadas por comas:

```
ORIGENES_PERMITIDOS=https://tu-app.vercel.app
```

Comando de arranque:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```
