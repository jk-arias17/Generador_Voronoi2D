## 0. LIBRERÍAS
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy.spatial import Voronoi, cKDTree
import math
import random
import time

## 1. PÁRAMETROS

B = 150 # Base del dominio
H = 150 # Altura del dominio
dominio = box(0, 0, B, H) # Creación del dominio
area_dominio = B * H  # Área total del dominio

# MODIFICAR
porc_agregados_total = 0.85 # % de área del dominio que será agregado

# --- EXPORTACIÓN VEM (TXT) ---
exportar_txts_vem_activo = True
nombre_txt_nodos = "Input_nodos.txt"
nombre_txt_conectividad = "Input_conectividad.txt"
# Redondeo (en mm) para unificar nodos/segmentos
NDIGITS_SNAP = 6           # 1e-6 mm
# Filtro de polígonos muy pequeños (mm^2)
AREA_MIN_MM2 = 1e-8
# Si quieres probar patch test con material uniforme, pon True (todo code=5)
FORZAR_MATERIAL_UNIFORME = False

fraccion_clases = np.array([0.10, 0.35, 0.40, 0.15])  # [3/4, 1/2, 3/8, #4]
area_agregados_objetivo = porc_agregados_total * area_dominio
areas_obj_granu = fraccion_clases * area_agregados_objetivo

# Rangos de diámetros (mm) por granulometría 
diametro_34 = (13.0, 19.0)
diametro_12 = (9.5, 12.5)
diametro_38 = (4.75, 9.5)
diametro_4  = (2.36, 4.75)

def rango_area_desde_diametro(dmin, dmax):
    """Función para calcular el área de la partícula considerando que es circular"""
    return math.pi*(dmin/2)**2, math.pi*(dmax/2)**2

# Cálculo de áreas mínimas, máximas y media
area34_min, area34_max = rango_area_desde_diametro(*diametro_34)
area12_min, area12_max = rango_area_desde_diametro(*diametro_12)
area38_min, area38_max = rango_area_desde_diametro(*diametro_38)
area4_min,  area4_max  = rango_area_desde_diametro(*diametro_4)
areas_min = np.array([area34_min, area12_min, area38_min, area4_min])
areas_max = np.array([area34_max, area12_max, area38_max, area4_max])
areas_media = 0.5 * (areas_min + areas_max)

tolerancia = 0.02 # Tolerancia para cumplir los objetivos


semilla = None
chequeo_radio = {
    1: 22.0,  # 3/4"
    2: 18.0,  # 1/2"
    3: 14.0,  # 3/8"
    4: 10.0,  # #4 
}
chequeofuerte_radio = {k: 0.70*v for k, v in chequeo_radio.items()}
MIN_RADIO_FACTOR = 0.65  
MIN_GAP_FACTOR = 0.55   
RELAX_RADIO_STEP = 0.90   
RELAX_GAP_STEP = 0.88   

# Espesores mínimos de matriz
espesor_min_34 = 0.40
espesor_min_12 = 0.35
espesor_min_38 = 0.25
espesor_min_4  = 0.12
espesores_min_granu = {1: espesor_min_34, 2: espesor_min_12, 3: espesor_min_38, 4: espesor_min_4}

area_objetivo_matriz = 15 # Área matriz por granulometría

#Límite de puntos y celdas
max_puntos_voronoi = 12000
max_celdas_voronoi = 30000

area4_media = float(areas_media[3])

## 2. FUNCIONES (DEF)

def validacion_geom(geom):
    """
    Función para validar geometrías
    """
    # Vacío, no se hace nada
    if geom is None or geom.is_empty:
        return geom
    
    # Se trata de arreglar la geometría
    try:
        from shapely.validation import make_valid
        g2 = make_valid(geom)
        return g2 if (g2 is not None and not g2.is_empty) else geom
    
    except Exception:
    # Si no sirve lo anterior, se trata de arreglar de otra manera
        try:
            g2 = geom.buffer(0)
            return g2 if (g2 is not None and not g2.is_empty) else geom
    # Si no, se retorna el original
        except Exception:
            return geom

def separar_poligonos(geom):
    """
    Función que convierte Polygon/MultiPolygon en listas y además filtra piezas  muy pequeñas.
    """
    # Se asegura de que la geometría sea válida
    geom = validacion_geom(geom)

    # Si esta vacío, se retorna vacío
    if geom is None or geom.is_empty:
        return []
    
    if isinstance(geom, MultiPolygon):
        return [validacion_geom(g) for g in geom.geoms if (g and (not g.is_empty) and g.area > 1e-9)]
    return [geom] if geom.area > 1e-9 else []

def puntos_aleatorios_en_poligono(poligono: Polygon, n_puntos: int, max_intentos: int = 400000):
    """
    Función para la generación de puntos en el bounding box y acepta solo los que caen dentro del polígono
    """
    poligono = validacion_geom(poligono)
    if poligono is None or poligono.is_empty:
        return np.zeros((0, 2))

    minx, miny, maxx, maxy = poligono.bounds
    puntos = []
    intentos = 0

    while len(puntos) < n_puntos and intentos < max_intentos:
        intentos += 1
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if poligono.contains(Point(x, y)):
            puntos.append((x, y))

    return np.array(puntos)

def puntos_aleatorios_en_geometria(geom, n_puntos: int):
    """
    Función que distribuye los puntos proporcional al área de cada componente.
    """
    geom = validacion_geom(geom)
    if geom is None or geom.is_empty or n_puntos <= 0:
        return np.zeros((0, 2))

    poligonos = separar_poligonos(geom)
    if not poligonos:
        return np.zeros((0, 2))

    areas = np.array([p.area for p in poligonos], dtype=float)
    probs = areas / areas.sum()
    conteos = np.random.multinomial(n_puntos, probs)

    lista_pts = []
    for p, k in zip(poligonos, conteos):
        if k <= 0:
            continue
        pts = puntos_aleatorios_en_poligono(p, k)
        if pts.shape[0] > 0:
            lista_pts.append(pts)

    return np.vstack(lista_pts) if lista_pts else np.zeros((0, 2))

def voronoi_recortado(poligonos_semilla: np.ndarray, recorte):
    """
    Construye Voronoi 2D y recorta cada celda a la geometría recorte.
    """
    recorte = validacion_geom(recorte)
    if recorte is None or recorte.is_empty or poligonos_semilla.shape[0] < 3:
        return []

    minx, miny, maxx, maxy = recorte.bounds
    L = max(maxx - minx, maxy - miny) * 10.0
    puntos_extra = np.array([
        [minx - L, miny - L],
        [minx - L, maxy + L],
        [maxx + L, miny - L],
        [maxx + L, maxy + L],
    ])

    puntos = np.vstack([poligonos_semilla, puntos_extra])
    vor = Voronoi(puntos)

    celdas = []
    N = poligonos_semilla.shape[0]

    for i in range(N):
        idx_region = vor.point_region[i]
        region = vor.regions[idx_region]
        # Si tiene -1, significa que es una región infinita
        if -1 in region or len(region) == 0:
            continue

        celda = validacion_geom(Polygon(vor.vertices[region]))
        if celda is None or celda.is_empty:
            continue

        celda_clip = validacion_geom(celda.intersection(recorte))
        if celda_clip is None or celda_clip.is_empty:
            continue

        if isinstance(celda_clip, MultiPolygon):
            celda_clip = max(celda_clip.geoms, key=lambda g: g.area)
            celda_clip = validacion_geom(celda_clip)

        if celda_clip is not None and (not celda_clip.is_empty) and celda_clip.area > 1e-9:
            celdas.append(celda_clip)

    return celdas



def construir_indice_celdas(celdas):
    """STRtree para ubicar rápidamente la celda que contiene un punto."""
    return STRtree(celdas)

def indice_celda_desde_punto(tree, celdas, x, y):
    """Retorna el índice de la celda que contiene (x,y). Compatible con Shapely 1.x/2.x."""
    p = Point(float(x), float(y))
    cand = tree.query(p)
    if len(cand) == 0:
        return None
    primero = cand[0]
    if hasattr(primero, "contains"):
        # Shapely 1.x: devuelve geometrías
        for g in cand:
            if g.contains(p):
                # candidatos son pocos: búsqueda lineal por identidad
                for i, cc in enumerate(celdas):
                    if cc is g:
                        return i
                # fallback por equals
                for i, cc in enumerate(celdas):
                    if cc.equals(g):
                        return i
        return None
    else:
        # Shapely 2.x: devuelve índices
        for idx in cand:
            ii = int(idx)
            if celdas[ii].contains(p):
                return ii
        return None

def generador_puntos_estratificados(B, H, nx=10, ny=10):
    """
    Generador infinito de puntos uniformes
    recorre una grilla nx*ny en orden aleatorio por ciclos.
"""
    while True:
        celdillas = [(i, j) for i in range(nx) for j in range(ny)]
        random.shuffle(celdillas)
        for i, j in celdillas:
            x0 = (i / nx) * B
            x1 = ((i + 1) / nx) * B
            y0 = (j / ny) * H
            y1 = ((j + 1) / ny) * H
            yield (random.uniform(x0, x1), random.uniform(y0, y1))

def inset_poligono(poligono, g):
    """
    Función para la creación de la matriz alrededor de partículas
    """
    poligono = validacion_geom(poligono)
    if poligono is None or poligono.is_empty:
        return None

    p = validacion_geom(poligono.buffer(-g, join_style=2))
    if p is None or p.is_empty:
        return None

    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda gg: gg.area)
        p = validacion_geom(p)

    return p if (p is not None and not p.is_empty and p.area > 1e-9) else None

def escalar_a_area(poligono, area_obj):
    """
    Función que escala un polígono alrededor de su centroide para aproximar el área deseada.
    """
    poligono = validacion_geom(poligono)
    if poligono is None or poligono.is_empty or poligono.area <= 1e-12:
        return None

    s = math.sqrt(area_obj / poligono.area)
    if s <= 0:
        return None

    from shapely import affinity
    p2 = validacion_geom(affinity.scale(poligono, xfact=s, yfact=s, origin="centroid"))

    return p2 if (p2 is not None and not p2.is_empty and p2.area > 1e-9) else None

def dividir_region_matriz(region_matriz, area_objetivo):
    """
    Función que subdivide una región de matriz en múltiples polígonos usando Voronoi interno:
    """
    region_matriz = validacion_geom(region_matriz)
    if region_matriz is None or region_matriz.is_empty:
        return []

    A = region_matriz.area
    n = int(round(A / max(area_objetivo, 1e-9)))
    n = max(3, min(max_puntos_voronoi, n))

    if n < 3:
        return separar_poligonos(region_matriz)

    pts = puntos_aleatorios_en_geometria(region_matriz, n)
    if pts.shape[0] < 3:
        return separar_poligonos(region_matriz)

    celdas = voronoi_recortado(pts, region_matriz)
    if len(celdas) > max_celdas_voronoi:
        celdas = random.sample(celdas, max_celdas_voronoi)

    union = validacion_geom(unary_union(celdas)) if celdas else None
    if union is not None and (not union.is_empty):
        huecos = validacion_geom(region_matriz.difference(union))
        celdas.extend(separar_poligonos(huecos))

    return celdas

def subdividir_celda(celda, area_obj_celda):
    """
    Función que subdivide una celda grande en subceldas (Voronoi interno) para #4
    """
    celda = validacion_geom(celda)
    if celda is None or celda.is_empty:
        return []

    n = int(round(celda.area / max(area_obj_celda, 1e-9)))
    n = max(3, min(250, n))

    pts = puntos_aleatorios_en_poligono(celda, n)
    if pts.shape[0] < 3:
        return [celda]

    return voronoi_recortado(pts, celda)

## 3. VORONOI

def generar_celdas_globales():
    """
    Función que genera el Voronoi global en el dominio.
    """
    area_promedio_agregado = float((fraccion_clases * areas_media).sum())
    N = int(round(area_agregados_objetivo / max(area_promedio_agregado, 1e-9)))
    N = max(250, N)

    semillas = np.zeros((N, 2))
    semillas[:, 0] = np.random.rand(N) * B
    semillas[:, 1] = np.random.rand(N) * H

    return voronoi_recortado(semillas, dominio)

## 4. CREACIÓN PARTÍCULAS 

def asignacion_granu_celdas(celdas, indices_libres, id_clase, area_objetivo_clase, tree, punto_gen,
                            centros_clase, radio_check, radio_factor=1.0, gap_factor=1.0):
    """
    Enfoque del profesor + verificación por radio (anti-agrupación).
    """
    area_min = areas_min[id_clase - 1]
    area_max = areas_max[id_clase - 1]
    gap = espesores_min_granu[id_clase] * float(gap_factor)
    area_acum = 0.0
    agregados = {}
    usadas = set()
    libres = set(indices_libres)

    max_intentos = max(700000, 18000 * max(1, len(indices_libres)))
    rmin = float(radio_check.get(id_clase, 0.0)) * float(radio_factor)
    r_hard = float(chequeofuerte_radio.get(id_clase, 0.0))
    pts = np.array(centros_clase, dtype=float) if len(centros_clase) > 0 else np.zeros((0, 2), dtype=float)
    kdt = cKDTree(pts) if pts.shape[0] > 0 else None
    rebuild_every = 25
    added_since_rebuild = 0

    intentos = 0
    while (area_acum < area_objetivo_clase * (1 - tolerancia)) and (intentos < max_intentos) and libres:
        intentos += 1
        x, y = next(punto_gen)

        idx = indice_celda_desde_punto(tree, celdas, x, y)
        if idx is None or idx not in libres:
            continue

        if kdt is not None and (r_hard > 0.0 or rmin > 0.0):
            cxy = celdas[idx].centroid
            cx, cy = float(cxy.x), float(cxy.y)

            if r_hard > 0.0:
                if len(kdt.query_ball_point([cx, cy], r=r_hard)) > 0:
                    continue

            if rmin > 0.0:
                if len(kdt.query_ball_point([cx, cy], r=rmin)) > 0:
                    continue
        # ----------------------------------------------
        pin = inset_poligono(celdas[idx], gap)
        if pin is None:
            continue

        area_deseada = min(area_max, pin.area * 0.98)
        if area_deseada < area_min:
            continue

        restante = area_objetivo_clase - area_acum
        if restante < area_deseada:
            if restante >= area_min:
                area_deseada = restante
            else:
                break

        agregado = escalar_a_area(pin, area_deseada)
        if agregado is None:
            continue

        agregados[idx] = agregado
        usadas.add(idx)
        libres.remove(idx)
        area_acum += agregado.area

        cxy = celdas[idx].centroid
        centros_clase.append((float(cxy.x), float(cxy.y)))
        added_since_rebuild += 1
        if added_since_rebuild >= rebuild_every:
            pts = np.array(centros_clase, dtype=float)
            kdt = cKDTree(pts) if pts.shape[0] > 0 else None
            added_since_rebuild = 0

    return agregados, usadas, area_acum
## 5. CREACIÓN PARTÍCULAS #4

def construir_celdas_para_4(celdas, indices_libres):
    """
    Función para la creación de partículas #4
    """

    area_obj_celda_4 = (area4_media / porc_agregados_total)

    salida = []
    for i in indices_libres:
        celda = celdas[i]
        if celda.area > 3 * area_obj_celda_4:
            salida.extend(subdividir_celda(celda, area_obj_celda_4))
        else:
            salida.append(celda)

    return salida

def asignacion_4_a_celdas(celdas_4, area_objetivo_4):
    """
    Función que asigna agregados #4 dentro de celdas 
    """
    gap = espesores_min_granu[4]
    area_min = area4_min
    area_max = area4_max

    area_acum = 0
    lista_agregados_4 = []
    lista_matriz = []

    celdas_4 = sorted(celdas_4, key=lambda p: -p.area)

    for celda in celdas_4:
        if area_acum >= area_objetivo_4 * (1 - tolerancia):
            lista_matriz.extend(dividir_region_matriz(celda, area_objetivo_matriz))
            continue
        pin = inset_poligono(celda, gap)
        if pin is None:
            lista_matriz.extend(dividir_region_matriz(celda, area_objetivo_matriz))
            continue

        area_deseada = min(area_max, pin.area * 0.98)
        if area_deseada < area_min:
            lista_matriz.extend(dividir_region_matriz(celda, area_objetivo_matriz))
            continue

        restante = area_objetivo_4 - area_acum
        if restante < area_deseada:
            if restante >= area_min:
                area_deseada = restante
            else:
                lista_matriz.extend(dividir_region_matriz(celda, area_objetivo_matriz))
                continue

        agregado_4 = escalar_a_area(pin, area_deseada)
        if agregado_4 is None:
            lista_matriz.extend(dividir_region_matriz(celda, area_objetivo_matriz))
            continue

        lista_agregados_4.append(agregado_4)
        area_acum += agregado_4.area

        # anillo de matriz de esta celda: celda - agregado_4
        anillo = validacion_geom(celda.difference(agregado_4))
        lista_matriz.extend(dividir_region_matriz(anillo, area_objetivo_matriz))

    return lista_agregados_4, lista_matriz, area_acum

## 6. CREACIÓN MATRIZ

def construir_matriz_macro(celdas, agregados_dict, area_objetivo):
    """
    Para cada agregado macro ubicado en una celda:
      - anillo = celda - agregado
      - subdividir ese anillo en muchos polígonos de matriz
    """
    partes_matriz = []
    for idx_celda, agregado in agregados_dict.items():
        anillo = validacion_geom(celdas[idx_celda].difference(agregado))
        partes_matriz.extend(dividir_region_matriz(anillo, area_objetivo))
    return partes_matriz

## 7. PLOT Y RESUMEN

def graficar_microestructura(poligonos_matriz, agg_34, agg_12, agg_38, agg_4):
    colores = {
        "matriz": (0.85, 0.85, 0.85),
        1: (1.00, 0.30, 0.30),  # 3/4
        2: (0.20, 0.40, 1.00),  # 1/2
        3: (0.20, 0.80, 0.20),  # 3/8
        4: (1.00, 1.00, 0.00),  # #4
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    for p in poligonos_matriz:
        p = validacion_geom(p)
        if p is None or p.is_empty:
            continue
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=colores["matriz"], edgecolor="k", linewidth=0.01, antialiased=False)

    for p in agg_34:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=colores[1], edgecolor="k", linewidth=0.01, antialiased=False)

    for p in agg_12:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=colores[2], edgecolor="k", linewidth=0.01, antialiased=False)

    for p in agg_38:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=colores[3], edgecolor="k", linewidth=0.01, antialiased=False)

    for p in agg_4:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=colores[4], edgecolor="k", linewidth=0.01, antialiased=False)

    ax.set_aspect("equal", "box")
    ax.set_xlim(0, B); ax.set_ylim(0, H)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Diagrama Voronoi 2D")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=colores[1], label='3/4"'),
        Patch(color=colores[2], label='1/2"'),
        Patch(color=colores[3], label='3/8"'),
        Patch(color=colores[4], label='#4'),
        Patch(color=colores["matriz"], label='Matriz'),
    ], loc="upper right")

    plt.tight_layout()
    plt.show()

def resumen_areas(agregados_34, agregados_12, agregados_38, lista_agregados_4):

    A34 = sum(p.area for p in agregados_34) if agregados_34 else 0.0
    A12 = sum(p.area for p in agregados_12) if agregados_12 else 0.0
    A38 = sum(p.area for p in agregados_38) if agregados_38 else 0.0
    A4  = sum(p.area for p in lista_agregados_4) if lista_agregados_4 else 0.0

    A_aggs = A34 + A12 + A38 + A4
    A_mat = area_dominio - A_aggs

    print("Resumen:")
    print(f"Área dominio            = {area_dominio:10.2f} mm^2")
    print(f"Objetivo agregados      = {area_agregados_objetivo:10.2f} mm^2 ({100*area_agregados_objetivo/area_dominio:6.2f}%)")
    print(f"Agregados logrados      = {A_aggs:10.2f} mm^2 ({100*A_aggs/area_dominio:6.2f}%)")
    print(f"Matriz lograda          = {A_mat:10.2f} mm^2 ({100*A_mat/area_dominio:6.2f}%)")
    print("")
    print("")
    print("Áreas reales:")
    if A_aggs > 0:
        print(f'  3/4"  = {A34:10.2f} mm^2 ({100*A34/A_aggs:6.2f}%)')
        print(f'  1/2"  = {A12:10.2f} mm^2 ({100*A12/A_aggs:6.2f}%)')
        print(f'  3/8"  = {A38:10.2f} mm^2 ({100*A38/A_aggs:6.2f}%)')
        print(f'  #4    = {A4 :10.2f} mm^2 ({100*A4 /A_aggs:6.2f}%)')



# =========================
# EXPORTADOR VEM CONFORMING (fix de T-junctions directo)
# =========================

# Tolerancia para detectar vértices sobre una arista (en mm)
SNAP_TOL_MM = 1e-4

def _snap(x, y, nd=NDIGITS_SNAP):
    """Redondea un punto a nd decimales (en mm)."""
    return (round(float(x), nd), round(float(y), nd))

def _ring_coords_snapped(ring, nd=NDIGITS_SNAP):
    """
    Extrae coordenadas del anillo con snap, sin cierre repetido
    y sin duplicados consecutivos.
    """
    pts = [_snap(x, y, nd) for x, y in ring.coords]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    out = []
    for p in pts:
        if not out or p != out[-1]:
            out.append(p)
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out

def _signed_area2(coords):
    """Doble del área con signo: positivo = CCW, negativo = CW."""
    n = len(coords)
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def _pt_on_seg(vx, vy, ax, ay, bx, by, tol):
    """
    True si (vx,vy) cae ESTRICTAMENTE entre (ax,ay) y (bx,by)
    con tolerancia perpendicular `tol` (mm).
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < tol * tol:
        return False
    seg_len = math.sqrt(len2)
    # distancia perpendicular al segmento
    cross = dx * (vy - ay) - dy * (vx - ax)
    if abs(cross) > tol * seg_len:
        return False
    # posición paramétrica t ∈ (0, 1) estricto
    dot = (vx - ax) * dx + (vy - ay) * dy
    t = dot / len2
    margin = tol / seg_len
    return margin < t < 1.0 - margin

def _fix_tjunctions(code_coords_list, tol=SNAP_TOL_MM):
    """
    Para cada polígono, recorre sus aristas e inserta los vértices
    globales que caigan sobre ellas (T-junctions fix).
    Garantiza orientación CCW al final.
    Devuelve list of (code, [(x,y), ...]).
    """
    # Recoger todos los vértices únicos de todos los polígonos
    all_verts_set = set()
    for _, coords in code_coords_list:
        all_verts_set.update(coords)
    all_verts = list(all_verts_set)
    verts_arr = np.array(all_verts, dtype=float)
    kd = cKDTree(verts_arr)

    result = []
    for code, coords in code_coords_list:
        n = len(coords)
        new_coords = []

        for i in range(n):
            ax, ay = coords[i]
            bx, by = coords[(i + 1) % n]
            new_coords.append((ax, ay))

            dx, dy = bx - ax, by - ay
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len < tol:
                continue

            # Buscar vértices candidatos en la vecindad del segmento
            cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
            indices = kd.query_ball_point([cx, cy], seg_len / 2.0 + tol)

            intermedios = []
            for idx in indices:
                vx, vy = all_verts[idx]
                if (vx, vy) == (ax, ay) or (vx, vy) == (bx, by):
                    continue
                if _pt_on_seg(vx, vy, ax, ay, bx, by, tol):
                    t = ((vx - ax) * dx + (vy - ay) * dy) / (seg_len * seg_len)
                    intermedios.append((t, (vx, vy)))

            intermedios.sort()
            for _, v in intermedios:
                new_coords.append(v)

        # Limpiar duplicados consecutivos
        clean = []
        for v in new_coords:
            if not clean or v != clean[-1]:
                clean.append(v)
        if len(clean) >= 2 and clean[0] == clean[-1]:
            clean = clean[:-1]

        if len(clean) < 3:
            continue

        # Garantizar orientación CCW
        if _signed_area2(clean) < 0:
            clean = clean[::-1]

        result.append((code, clean))

    return result

def exportar_txts_vem_conforming(poligonos_matriz, agg_34, agg_12, agg_38, agg_4,
                                 nombre_nodos=nombre_txt_nodos,
                                 nombre_conect=nombre_txt_conectividad,
                                 ndigits=NDIGITS_SNAP):
    """
    Exporta Input_nodos.txt e Input_conectividad.txt con malla conforming.
    Estrategia:
      1. true_matrix = dominio − union(agregados)  → sin solapamientos por construcción
      2. Subdivisión de matriz con area_target < min(area agregados) → sin celdas con agujeros
      3. Fix de T-junctions para fronteras entre celdas de matriz
    - Orientación CCW garantizada  |  Coordenadas en metros
    - GDL X: 0..N-1  |  GDL Y: N..2N-1
    """
    # 1) Recopilar agregados con su código de material
    flat_aggs = []
    for code, lst in [(1, agg_34), (2, agg_12), (3, agg_38), (4, agg_4)]:
        if not lst:
            continue
        for p in lst:
            p = validacion_geom(p)
            if p is None or p.is_empty or p.area < AREA_MIN_MM2:
                continue
            if isinstance(p, MultiPolygon):
                for g in p.geoms:
                    g = validacion_geom(g)
                    if g and not g.is_empty and g.area >= AREA_MIN_MM2:
                        flat_aggs.append((code, g))
            else:
                flat_aggs.append((code, p))

    # 2) Calcular true_matrix = dominio − union(agregados)
    #    Garantía: ninguna celda de matriz se solapa con ningún agregado
    if flat_aggs:
        union_aggs = validacion_geom(unary_union([p for _, p in flat_aggs]))
        true_matrix = validacion_geom(dominio.difference(union_aggs)) if union_aggs else dominio
    else:
        true_matrix = dominio

    # 3) area_target < área del agregado más pequeño
    #    Garantía: ninguna celda Voronoi es suficientemente grande para contener
    #    un agregado entero → celdas sin agujeros internos
    if flat_aggs:
        min_agg_area = min(p.area for _, p in flat_aggs)
        area_target = min(area_objetivo_matriz, min_agg_area * 0.75)
    else:
        area_target = area_objetivo_matriz

    # 4) Subdivisión de la región de matriz
    print(f"[INFO] Subdividiendo matriz (area_target={area_target:.2f} mm²)...")
    matrix_cells_raw = []
    if true_matrix and not true_matrix.is_empty:
        matrix_cells_raw = dividir_region_matriz(true_matrix, area_target)

    # 5) Filtrar y gestionar celdas residuales con agujeros (safety net)
    matrix_cells = []
    for cell in matrix_cells_raw:
        cell = validacion_geom(cell)
        if cell is None or cell.is_empty or cell.area < AREA_MIN_MM2:
            continue
        if list(cell.interiors):
            # Celda con agujero residual: subdividir de nuevo con área menor
            sub = dividir_region_matriz(cell, cell.area / max(4, len(list(cell.interiors)) + 3))
            for sc in sub:
                sc = validacion_geom(sc)
                if sc and not sc.is_empty and sc.area >= AREA_MIN_MM2 and not list(sc.interiors):
                    matrix_cells.append(sc)
        else:
            matrix_cells.append(cell)

    print(f"[INFO] Agregados: {len(flat_aggs)}  |  Celdas de matriz: {len(matrix_cells)}")

    # 6) Reunir todos los polígonos con código de material y hacer snap
    snapped = []
    for code, poly in flat_aggs:
        coords = _ring_coords_snapped(poly.exterior, ndigits)
        if len(coords) >= 3:
            snapped.append((code, coords))
    for cell in matrix_cells:
        coords = _ring_coords_snapped(cell.exterior, ndigits)
        if len(coords) >= 3:
            snapped.append((5, coords))

    if not snapped:
        print("[ERROR] No hay polígonos para exportar.")
        return

    print(f"[INFO] Polígonos antes del fix: {len(snapped)}")

    # 7) Fix de T-junctions entre celdas de matriz adyacentes
    conforming = _fix_tjunctions(snapped, tol=SNAP_TOL_MM)

    print(f"[INFO] Polígonos después del fix: {len(conforming)}")

    # 4) Construir mapa de nodos global (coordenada → id único)
    node_map = {}
    node_xy  = []

    def get_nid(xy):
        if xy not in node_map:
            node_map[xy] = len(node_xy)
            node_xy.append(xy)
        return node_map[xy]

    elems = []
    for code, coords in conforming:
        ids = [get_nid(v) for v in coords]
        # Eliminar duplicados consecutivos (por si el snap los generó)
        clean_ids = []
        for nid in ids:
            if not clean_ids or nid != clean_ids[-1]:
                clean_ids.append(nid)
        if len(clean_ids) >= 2 and clean_ids[0] == clean_ids[-1]:
            clean_ids = clean_ids[:-1]
        if len(set(clean_ids)) < 3:
            continue
        elems.append((code, clean_ids))

    # 5) Validación y corrección de aristas no-manifold
    from collections import Counter as _Counter

    def _build_edge_map(elems_list):
        ec = {}
        for eid, (_, conn) in enumerate(elems_list):
            m = len(conn)
            for j in range(m):
                a = conn[j]
                b = conn[(j + 1) % m]
                key = (min(a, b), max(a, b))
                if key not in ec:
                    ec[key] = []
                ec[key].append(eid)
        return ec

    def _elem_area(conn):
        coords = [node_xy[nid] for nid in conn]
        return abs(_signed_area2(coords))

    # Iteración de limpieza: elimina elementos slivers causantes de no-manifold
    for _pass in range(10):
        edge_map = _build_edge_map(elems)
        nm_edges = {k: v for k, v in edge_map.items() if len(v) > 2}
        if not nm_edges:
            break
        to_remove = set()
        for edge, eids in nm_edges.items():
            # El elemento más pequeño que toca esta arista es el artefacto
            culprit = min(eids, key=lambda e: _elem_area(elems[e][1]))
            to_remove.add(culprit)
        elems = [e for i, e in enumerate(elems) if i not in to_remove]
        print(f"[FIX] Pasada {_pass + 1}: eliminados {len(to_remove)} elementos no-manifold.")

    # Validación final
    edge_cnt = {}
    for _, conn in elems:
        m = len(conn)
        for j in range(m):
            a = conn[j]
            b = conn[(j + 1) % m]
            key = (min(a, b), max(a, b))
            edge_cnt[key] = edge_cnt.get(key, 0) + 1
    dist = _Counter(edge_cnt.values())
    bad  = sum(1 for v in edge_cnt.values() if v > 2)
    print(f"[CHECK] Aristas por #elem: {dict(dist)}  |  no-manifold (>2): {bad}")
    if bad == 0:
        print("[OK] Malla conforming sin aristas no-manifold.")
    else:
        print(f"[WARN] Quedaron {bad} aristas no-manifold tras la limpieza.")

    # 6) Escribir Input_nodos.txt  (coordenadas en METROS)
    N = len(node_xy)
    with open(nombre_nodos, "w", encoding="utf-8") as f:
        for i, (xmm, ymm) in enumerate(node_xy):
            f.write(f"{i}\t{i + N}\t{xmm / 1000.0:.9f}\t{ymm / 1000.0:.9f}\t0\t0\t0\t0\n")

    # 7) Escribir Input_conectividad.txt
    with open(nombre_conect, "w", encoding="utf-8") as f:
        for eid, (code, conn) in enumerate(elems):
            f.write("\t".join(map(str, [eid, code, code] + conn)) + "\n")

    print(f"[OK] Exportados {N} nodos  |  {len(elems)} elementos")
    print(f"[OK] Archivos: {nombre_nodos},  {nombre_conect}")


def main():
    # reproducibilidad
    if semilla is None:
        SEED_LOCAL = int(time.time() * 1000) % (2**32 - 1)
    else:
        SEED_LOCAL = int(semilla)
    np.random.seed(SEED_LOCAL)
    random.seed(SEED_LOCAL)

    # 1) celdas Voronoi globales
    celdas = generar_celdas_globales()
    indices_libres = list(range(len(celdas)))

    tree_celdas = construir_indice_celdas(celdas)
    punto_gen = generador_puntos_estratificados(B, H, nx=12, ny=12)

    centros_por_clase = {1: [], 2: [], 3: [], 4: []}
    # 2) asignación macro por orden (3/4, 1/2, 3/8) con relajación (menos matriz)
    def asignar_clase_con_relajacion(id_clase, area_obj):
        radio_factor = 1.0
        gap_factor = 1.0
        agg_total = {}
        usadas_total = set()
        area_total = 0.0

        
        while True:
            area_restante = max(0.0, area_obj - area_total)
            if area_restante <= area_obj * 1e-6:
                break

            agg, usadas, area_lograda = asignacion_granu_celdas(
                celdas, indices_libres, id_clase, area_restante,
                tree_celdas, punto_gen,
                centros_por_clase[id_clase], chequeo_radio,
                radio_factor=radio_factor, gap_factor=gap_factor
            )

            agg_total.update(agg)
            usadas_total |= usadas
            area_total += area_lograda

            # Evitar sobrepasar el target por tolerancias numéricas
            if area_total > area_obj:
                area_total = area_obj

            for u in usadas:
                if u in indices_libres:
                    indices_libres.remove(u)

            if area_total >= area_obj * (1 - tolerancia):
                break
            if len(indices_libres) == 0:
                break

            radio_factor *= RELAX_RADIO_STEP
            gap_factor *= RELAX_GAP_STEP

            if radio_factor < MIN_RADIO_FACTOR and gap_factor < MIN_GAP_FACTOR:
                break

            radio_factor = max(radio_factor, MIN_RADIO_FACTOR)
            gap_factor = max(gap_factor, MIN_GAP_FACTOR)

        return agg_total, usadas_total, area_total

    agg_34_dict, usadas_34, _ = asignar_clase_con_relajacion(1, areas_obj_granu[0])
    agg_12_dict, usadas_12, _ = asignar_clase_con_relajacion(2, areas_obj_granu[1])
    agg_38_dict, usadas_38, _ = asignar_clase_con_relajacion(3, areas_obj_granu[2])

    # 3) construir matriz macro (anillos por celda, subdivididos)
    poligonos_matriz = []
    poligonos_matriz.extend(construir_matriz_macro(celdas, agg_34_dict, area_objetivo_matriz))
    poligonos_matriz.extend(construir_matriz_macro(celdas, agg_12_dict, area_objetivo_matriz))
    poligonos_matriz.extend(construir_matriz_macro(celdas, agg_38_dict, area_objetivo_matriz))

    # 4) preparar celdas para #4 (subdividir celdas grandes)
    celdas_para_4 = construir_celdas_para_4(celdas, indices_libres)

    # 5) asignar #4 y su matriz
    lista_agregados_4, matriz_4, _ = asignacion_4_a_celdas(celdas_para_4, areas_obj_granu[3])
    poligonos_matriz.extend(matriz_4)

    # 6) sellar huecos numéricos: dominio - unión(de todo)
    todos = list(agg_34_dict.values()) + list(agg_12_dict.values()) + list(agg_38_dict.values()) + lista_agregados_4 + poligonos_matriz
    union_todo = validacion_geom(unary_union(todos)) if todos else None

    huecos = validacion_geom(dominio.difference(union_todo)) if (union_todo is not None and not union_todo.is_empty) else None
    poligonos_matriz.extend(separar_poligonos(huecos))

    # 7) preparar listas para plot y resumen
    lista_34 = list(agg_34_dict.values())
    lista_12 = list(agg_12_dict.values())
    lista_38 = list(agg_38_dict.values())

    # 8) plot + resumen
    
    # 8) exportar TXT VEM (malla CONFORMING)
    if exportar_txts_vem_activo:
        exportar_txts_vem_conforming(
            poligonos_matriz, lista_34, lista_12, lista_38, lista_agregados_4,
            nombre_nodos=nombre_txt_nodos, nombre_conect=nombre_txt_conectividad,
            ndigits=NDIGITS_SNAP
        )
    graficar_microestructura(poligonos_matriz, lista_34, lista_12, lista_38, lista_agregados_4)
    resumen_areas(lista_34, lista_12, lista_38, lista_agregados_4)

if __name__ == "__main__":
    main()

