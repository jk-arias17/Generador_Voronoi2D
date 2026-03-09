# 0. Librerías
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy.spatial import Voronoi, cKDTree
import math
import random
import time
from shapely.ops import polygonize as _polygonize

# 1. Parámetros del dominio y la mezcla

# MODIFICAR
B = 150  # ancho del dominio en mm
H = 150  # alto del dominio en mm
fraccion_agregados = 0.4 # Porcentaje de área total que será agregado
area_celda_matriz = 5 # Área objetivo de matriz (mm^2)

dominio = box(0, 0, B, H)
area_dominio = B * H

# Activar exportación de archivos para VEM
exportar_txts = True
archivo_nodos = "Input_nodos.txt"
archivo_conectividad = "Input_conectividad.txt"

# Precisión de redondeo para unificar nodos (en mm). 
precision_nodos = 4
# Área mínima para considerar un polígono válido (descarta sliver residuales)
area_minima = 1e-8
# Para testear con un solo material, poner True (todo sale como código 5)
material_uniforme = False

# Fracción de cada granulometría sobre el total de agregados: [3/4", 1/2", 3/8", #4]
fracciones = np.array([0.10, 0.35, 0.40, 0.15])
area_total_agg = fraccion_agregados * area_dominio
area_por_clase = fracciones * area_total_agg

# Rangos de diámetros por tamiz (mm)
diametro_34 = (13.0, 19.0)
diametro_12 = (9.5, 12.5)
diametro_38 = (4.75, 9.5)
diametro_4  = (2.36, 4.75)

def area_desde_diametro(dmin, dmax):
    # Asume partícula circular
    return math.pi*(dmin/2)**2, math.pi*(dmax/2)**2

area34_min, area34_max = area_desde_diametro(*diametro_34)
area12_min, area12_max = area_desde_diametro(*diametro_12)
area38_min, area38_max = area_desde_diametro(*diametro_38)
area4_min,  area4_max  = area_desde_diametro(*diametro_4)
areas_min = np.array([area34_min, area12_min, area38_min, area4_min])
areas_max = np.array([area34_max, area12_max, area38_max, area4_max])
areas_media = 0.5 * (areas_min + areas_max)

# Tolerancia para considerar que se alcanzó el área objetivo (2%)
tolerancia = 0.02

semilla = None  # None = aleatoria cada vez; poner un número para reproducir resultados

# Radio de separación mínimo entre centros de partículas del mismo tipo
radio_separacion = {
    1: 22.0,  # 3/4"
    2: 18.0,  # 1/2"
    3: 14.0,  # 3/8"
    4: 10.0,  # #4
}
# Radio duro: separación mínima garantizada sin relajar
radio_minimo = {k: 0.70*v for k, v in radio_separacion.items()}

# Factores mínimos de relajación (cuando el dominio ya está muy lleno)
min_radio_factor = 0.65
min_gap_factor = 0.55
paso_radio = 0.90
paso_gap = 0.88

# Espesor mínimo de capa de matriz alrededor de cada partícula
espesor_min_34 = 0.40
espesor_min_12 = 0.35
espesor_min_38 = 0.25
espesor_min_4  = 0.12
espesor_min = {1: espesor_min_34, 2: espesor_min_12, 3: espesor_min_38, 4: espesor_min_4}

# Área objetivo de cada celda de matriz (mm^2).


# Límites para no colapsar con dominios muy grandes
max_puntos = 12000
max_celdas = 30000

area_media_4 = float(areas_media[3])


# 2. Funciones auxiliares de geometría

def reparar(geom):
    # Intenta reparar la geometría si es inválida (self-intersections, etc.)
    if geom is None or geom.is_empty:
        return geom
    try:
        from shapely.validation import make_valid
        g2 = make_valid(geom)
        return g2 if (g2 is not None and not g2.is_empty) else geom
    except Exception:
        try:
            g2 = geom.buffer(0)
            return g2 if (g2 is not None and not g2.is_empty) else geom
        except Exception:
            return geom

def separar(geom):
    # Si viene un MultiPolygon lo separa en lista; filtra piezas insignificantes
    geom = reparar(geom)
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, MultiPolygon):
        return [reparar(g) for g in geom.geoms if (g and (not g.is_empty) and g.area > 1e-9)]
    return [geom] if geom.area > 1e-9 else []

def puntos_en_poligono(poligono: Polygon, n: int, max_intentos: int = 400000):
    # Muestreo por rechazo: genera puntos en el bounding box y acepta los que caen dentro
    poligono = reparar(poligono)
    if poligono is None or poligono.is_empty:
        return np.zeros((0, 2))
    minx, miny, maxx, maxy = poligono.bounds
    puntos = []
    intentos = 0
    while len(puntos) < n and intentos < max_intentos:
        intentos += 1
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if poligono.contains(Point(x, y)):
            puntos.append((x, y))
    return np.array(puntos)

def puntos_en_geom(geom, n: int):
    # Distribuye n puntos entre las piezas de la geometría proporcionalmente a su área
    geom = reparar(geom)
    if geom is None or geom.is_empty or n <= 0:
        return np.zeros((0, 2))
    piezas = separar(geom)
    if not piezas:
        return np.zeros((0, 2))
    areas = np.array([p.area for p in piezas], dtype=float)
    probs = areas / areas.sum()
    conteos = np.random.multinomial(n, probs)
    lista_pts = []
    for p, k in zip(piezas, conteos):
        if k <= 0:
            continue
        pts = puntos_en_poligono(p, k)
        if pts.shape[0] > 0:
            lista_pts.append(pts)
    return np.vstack(lista_pts) if lista_pts else np.zeros((0, 2))

def voronoi_recorte(semillas: np.ndarray, region):
    # Voronoi estándar de scipy, recortado al polígono dado.
    # Se añaden 4 puntos fantasma muy alejados para que ninguna celda quede abierta.
    region = reparar(region)
    if region is None or region.is_empty or semillas.shape[0] < 3:
        return []
    minx, miny, maxx, maxy = region.bounds
    L = max(maxx - minx, maxy - miny) * 10.0
    extra = np.array([
        [minx - L, miny - L],
        [minx - L, maxy + L],
        [maxx + L, miny - L],
        [maxx + L, maxy + L],
    ])
    puntos = np.vstack([semillas, extra])
    vor = Voronoi(puntos)
    celdas = []
    N = semillas.shape[0]
    for i in range(N):
        idx_region = vor.point_region[i]
        reg = vor.regions[idx_region]
        if -1 in reg or len(reg) == 0:
            continue
        celda = reparar(Polygon(vor.vertices[reg]))
        if celda is None or celda.is_empty:
            continue
        clip = reparar(celda.intersection(region))
        if clip is None or clip.is_empty:
            continue
        if isinstance(clip, MultiPolygon):
            clip = max(clip.geoms, key=lambda g: g.area)
            clip = reparar(clip)
        if clip is not None and (not clip.is_empty) and clip.area > 1e-9:
            celdas.append(clip)
    return celdas


# 3. Índice espacial y generadores de puntos

def armar_indice(celdas):
    return STRtree(celdas)

def celda_en_punto(tree, celdas, x, y):
    # Busca qué celda contiene el punto. Funciona con Shapely 1.x y 2.x
    # (en 2.x el árbol devuelve índices, en 1.x devuelve geometrías).
    p = Point(float(x), float(y))
    cand = tree.query(p)
    if len(cand) == 0:
        return None
    primero = cand[0]
    if hasattr(primero, "contains"):
        # Shapely 1.x
        for g in cand:
            if g.contains(p):
                for i, cc in enumerate(celdas):
                    if cc is g:
                        return i
                for i, cc in enumerate(celdas):
                    if cc.equals(g):
                        return i
        return None
    else:
        # Shapely 2.x
        for idx in cand:
            ii = int(idx)
            if celdas[ii].contains(p):
                return ii
        return None

def puntos_grilla(B, H, nx=10, ny=10):
    # Genera puntos de forma que cubran el dominio más uniformemente que muestreo puro aleatorio.
    # Divide el dominio en una grilla nx*ny y sortea el orden de visita en cada ciclo.
    while True:
        celdillas = [(i, j) for i in range(nx) for j in range(ny)]
        random.shuffle(celdillas)
        for i, j in celdillas:
            x0 = (i / nx) * B
            x1 = ((i + 1) / nx) * B
            y0 = (j / ny) * H
            y1 = ((j + 1) / ny) * H
            yield (random.uniform(x0, x1), random.uniform(y0, y1))

def encoger(poligono, g):
    # Reduce el polígono hacia adentro en g mm (buffer negativo).
    # Sirve para dejar un espesor mínimo de matriz alrededor del agregado.
    poligono = reparar(poligono)
    if poligono is None or poligono.is_empty:
        return None
    p = reparar(poligono.buffer(-g, join_style=2))
    if p is None or p.is_empty:
        return None
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda gg: gg.area)
        p = reparar(p)
    return p if (p is not None and not p.is_empty and p.area > 1e-9) else None

def escalar(poligono, area_obj):
    # Escala el polígono desde su centroide para que quede con area_obj mm².
    poligono = reparar(poligono)
    if poligono is None or poligono.is_empty or poligono.area <= 1e-12:
        return None
    s = math.sqrt(area_obj / poligono.area)
    if s <= 0:
        return None
    from shapely import affinity
    p2 = reparar(affinity.scale(poligono, xfact=s, yfact=s, origin="centroid"))
    return p2 if (p2 is not None and not p2.is_empty and p2.area > 1e-9) else None

def subdividir_matriz(region, area_obj):
    # Subdivide la región de matriz en celdas usando Voronoi interno.
    # Cuantas celdas: redondea area_total / area_obj.
    # Al final rellena los huecos que queden sin cubrir.
    region = reparar(region)
    if region is None or region.is_empty:
        return []
    A = region.area
    n = int(round(A / max(area_obj, 1e-9)))
    n = max(3, min(max_puntos, n))
    if n < 3:
        return separar(region)
    pts = puntos_en_geom(region, n)
    if pts.shape[0] < 3:
        return separar(region)
    celdas = voronoi_recorte(pts, region)
    if len(celdas) > max_celdas:
        celdas = random.sample(celdas, max_celdas)
    union = reparar(unary_union(celdas)) if celdas else None
    if union is not None and (not union.is_empty):
        huecos = reparar(region.difference(union))
        celdas.extend(separar(huecos))
    return celdas

def subdividir_celda(celda, area_obj):
    # Para celdas Voronoi muy grandes, las parte en subceldas más pequeñas.
    # Se usa antes de asignar los agregados #4.
    celda = reparar(celda)
    if celda is None or celda.is_empty:
        return []
    n = int(round(celda.area / max(area_obj, 1e-9)))
    n = max(3, min(250, n))
    pts = puntos_en_poligono(celda, n)
    if pts.shape[0] < 3:
        return [celda]
    return voronoi_recorte(pts, celda)


# 4. Voronoi global

def voronoi_base():
    # Crea el Voronoi base sobre el dominio completo.
    # La cantidad de semillas se estima a partir del área promedio de un agregado.
    area_promedio = float((fracciones * areas_media).sum())
    N = int(round(area_total_agg / max(area_promedio, 1e-9)))
    N = max(250, N)
    semillas = np.zeros((N, 2))
    semillas[:, 0] = np.random.rand(N) * B
    semillas[:, 1] = np.random.rand(N) * H
    return voronoi_recorte(semillas, dominio)


# 5. Asignación de agregados 3/4", 1/2" y 3/8"

def asignar_granu(celdas, indices_libres, id_clase, area_obj_clase, tree, gen_puntos,
                  centros, radio_check, radio_factor=1.0, gap_factor=1.0):
    area_min = areas_min[id_clase - 1]
    area_max = areas_max[id_clase - 1]
    gap = espesor_min[id_clase] * float(gap_factor)
    area_acum = 0.0
    agregados = {}
    usadas = set()
    libres = set(indices_libres)

    max_intentos = max(700000, 18000 * max(1, len(indices_libres)))
    rmin = float(radio_check.get(id_clase, 0.0)) * float(radio_factor)
    r_duro = float(radio_minimo.get(id_clase, 0.0))
    pts = np.array(centros, dtype=float) if len(centros) > 0 else np.zeros((0, 2), dtype=float)
    kdt = cKDTree(pts) if pts.shape[0] > 0 else None
    rebuild_every = 25
    added = 0

    intentos = 0
    while (area_acum < area_obj_clase * (1 - tolerancia)) and (intentos < max_intentos) and libres:
        intentos += 1
        x, y = next(gen_puntos)

        idx = celda_en_punto(tree, celdas, x, y)
        if idx is None or idx not in libres:
            continue

        # Verificar separación mínima entre partículas
        if kdt is not None and (r_duro > 0.0 or rmin > 0.0):
            cxy = celdas[idx].centroid
            cx, cy = float(cxy.x), float(cxy.y)
            if r_duro > 0.0:
                if len(kdt.query_ball_point([cx, cy], r=r_duro)) > 0:
                    continue
            if rmin > 0.0:
                if len(kdt.query_ball_point([cx, cy], r=rmin)) > 0:
                    continue

        pin = encoger(celdas[idx], gap)
        if pin is None:
            continue

        area_deseada = min(area_max, pin.area * 0.98)
        if area_deseada < area_min:
            continue

        restante = area_obj_clase - area_acum
        if restante < area_deseada:
            if restante >= area_min:
                area_deseada = restante
            else:
                break

        agg = escalar(pin, area_deseada)
        if agg is None:
            continue

        agregados[idx] = agg
        usadas.add(idx)
        libres.remove(idx)
        area_acum += agg.area

        cxy = celdas[idx].centroid
        centros.append((float(cxy.x), float(cxy.y)))
        added += 1
        if added >= rebuild_every:
            pts = np.array(centros, dtype=float)
            kdt = cKDTree(pts) if pts.shape[0] > 0 else None
            added = 0

    return agregados, usadas, area_acum


# 6. Asignación de agregados #4

def preparar_celdas_4(celdas, indices_libres):
    # Las celdas sobrantes (no usadas por 3/4", 1/2" y 3/8") se subdividen
    # si son muy grandes para el tamaño del agregado #4.
    area_obj = (area_media_4 / fraccion_agregados)
    salida = []
    for i in indices_libres:
        celda = celdas[i]
        if celda.area > 3 * area_obj:
            salida.extend(subdividir_celda(celda, area_obj))
        else:
            salida.append(celda)
    return salida

def asignar_fino(celdas_4, area_obj_4):
    # Recorre las celdas de mayor a menor área e intenta colocar un agregado #4.
    # Si la celda es demasiado chica o ya se alcanzó el área objetivo, la celda
    # pasa directo a ser subdividida como matriz.
    gap = espesor_min[4]
    area_acum = 0
    lista_agg_4 = []
    lista_matriz = []
    celdas_4 = sorted(celdas_4, key=lambda p: -p.area)

    for celda in celdas_4:
        if area_acum >= area_obj_4 * (1 - tolerancia):
            lista_matriz.extend(subdividir_matriz(celda, area_celda_matriz))
            continue
        pin = encoger(celda, gap)
        if pin is None:
            lista_matriz.extend(subdividir_matriz(celda, area_celda_matriz))
            continue
        area_deseada = min(area4_max, pin.area * 0.98)
        if area_deseada < area4_min:
            lista_matriz.extend(subdividir_matriz(celda, area_celda_matriz))
            continue
        restante = area_obj_4 - area_acum
        if restante < area_deseada:
            if restante >= area4_min:
                area_deseada = restante
            else:
                lista_matriz.extend(subdividir_matriz(celda, area_celda_matriz))
                continue
        agg_4 = escalar(pin, area_deseada)
        if agg_4 is None:
            lista_matriz.extend(subdividir_matriz(celda, area_celda_matriz))
            continue
        lista_agg_4.append(agg_4)
        area_acum += agg_4.area
        # La parte de la celda que no es agregado se vuelve matriz
        anillo = reparar(celda.difference(agg_4))
        lista_matriz.extend(subdividir_matriz(anillo, area_celda_matriz))

    return lista_agg_4, lista_matriz, area_acum

def matriz_macro(celdas, aggs_dict, area_obj):
    # Para cada agregado macro, saca el anillo de la celda que lo rodea y lo subdivide.
    partes = []
    for idx_celda, agg in aggs_dict.items():
        anillo = reparar(celdas[idx_celda].difference(agg))
        partes.extend(subdividir_matriz(anillo, area_obj))
    return partes


# 7. Visualización

def graficar(poligonos_matriz, agg_34, agg_12, agg_38, agg_4):
    colores = {
        "matriz": (0.85, 0.85, 0.85),
        1: (1.00, 0.30, 0.30),  # 3/4"
        2: (0.20, 0.40, 1.00),  # 1/2"
        3: (0.20, 0.80, 0.20),  # 3/8"
        4: (1.00, 1.00, 0.00),  # #4
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    for p in poligonos_matriz:
        p = reparar(p)
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

def resumen(agg_34, agg_12, agg_38, agg_4):
    A34 = sum(p.area for p in agg_34) if agg_34 else 0.0
    A12 = sum(p.area for p in agg_12) if agg_12 else 0.0
    A38 = sum(p.area for p in agg_38) if agg_38 else 0.0
    A4  = sum(p.area for p in agg_4)  if agg_4  else 0.0
    total_agg = A34 + A12 + A38 + A4
    total_mat = area_dominio - total_agg
    print("Resumen:")
    print(f"Área dominio            = {area_dominio:10.2f} mm^2")
    print(f"Objetivo agregados      = {area_total_agg:10.2f} mm^2 ({100*area_total_agg/area_dominio:6.2f}%)")
    print(f"Agregados logrados      = {total_agg:10.2f} mm^2 ({100*total_agg/area_dominio:6.2f}%)")
    print(f"Matriz lograda          = {total_mat:10.2f} mm^2 ({100*total_mat/area_dominio:6.2f}%)")
    print("")
    print("Áreas reales:")
    if total_agg > 0:
        print(f'  3/4"  = {A34:10.2f} mm^2 ({100*A34/total_agg:6.2f}%)')
        print(f'  1/2"  = {A12:10.2f} mm^2 ({100*A12/total_agg:6.2f}%)')
        print(f'  3/8"  = {A38:10.2f} mm^2 ({100*A38/total_agg:6.2f}%)')
        print(f'  #4    = {A4 :10.2f} mm^2 ({100*A4 /total_agg:6.2f}%)')


# 8. Exportación de archivos para VEM

# Tolerancia para detectar vértices sobre una arista (mm)
tol_snap = 1e-4

def _area_con_signo(coords):
    # Doble del área con signo: positivo = CCW (sentido antihorario), negativo = CW
    n = len(coords)
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def _fix_tjunctions(code_coords_list, tol=tol_snap):
    # Detecta vértices de otros polígonos que caen sobre una arista (T-junction)
    # e inserta esos vértices en la arista para que la malla sea conforming.
    # La búsqueda de candidatos por arista está vectorizada con numpy.
    todos_verts = set()
    for _, coords in code_coords_list:
        todos_verts.update(coords)
    verts = list(todos_verts)
    verts_arr = np.array(verts, dtype=float)
    kd = cKDTree(verts_arr)

    result = []
    for code, coords in code_coords_list:
        n = len(coords)
        coords_arr = np.array(coords, dtype=float)
        next_arr  = np.roll(coords_arr, -1, axis=0)
        dx_all = next_arr[:, 0] - coords_arr[:, 0]
        dy_all = next_arr[:, 1] - coords_arr[:, 1]
        len2_all = dx_all ** 2 + dy_all ** 2
        seglen_all = np.sqrt(len2_all)

        inserciones = {}  # índice de arista → [(t, (x,y)), ...]

        for i in range(n):
            seg_len = seglen_all[i]
            if seg_len < tol:
                continue
            ax, ay = coords_arr[i]
            bx, by = next_arr[i]
            dx, dy = dx_all[i], dy_all[i]
            len2   = len2_all[i]

            cx, cy = (ax + bx) * 0.5, (ay + by) * 0.5
            idxs = kd.query_ball_point([cx, cy], seg_len * 0.5 + tol)
            if not idxs:
                continue

            cands = verts_arr[idxs]

            # Saltar los extremos del propio segmento
            not_a = ~((cands[:, 0] == ax) & (cands[:, 1] == ay))
            not_b = ~((cands[:, 0] == bx) & (cands[:, 1] == by))
            mask  = not_a & not_b
            if not np.any(mask):
                continue
            cands = cands[mask]

            # Distancia perpendicular al segmento
            cross = dx * (cands[:, 1] - ay) - dy * (cands[:, 0] - ax)
            perp_ok = np.abs(cross) <= tol * seg_len
            if not np.any(perp_ok):
                continue
            cands = cands[perp_ok]

            # Posición paramétrica sobre el segmento (tiene que estar entre los extremos)
            dot    = (cands[:, 0] - ax) * dx + (cands[:, 1] - ay) * dy
            t_arr  = dot / len2
            margin = tol / seg_len
            t_ok   = (t_arr > margin) & (t_arr < 1.0 - margin)
            if not np.any(t_ok):
                continue

            cands = cands[t_ok]
            t_vals = t_arr[t_ok]
            orden  = np.argsort(t_vals)
            inserciones[i] = [(t_vals[j], (float(cands[j, 0]), float(cands[j, 1])))
                              for j in orden]

        # Reconstruir la lista de coordenadas con las inserciones
        nuevas = []
        for i in range(n):
            nuevas.append(coords[i])
            if i in inserciones:
                for _, v in inserciones[i]:
                    nuevas.append(v)

        # Limpiar duplicados
        clean = []
        for v in nuevas:
            if not clean or v != clean[-1]:
                clean.append(v)
        if len(clean) >= 2 and clean[0] == clean[-1]:
            clean = clean[:-1]

        if len(clean) < 3:
            continue

        # Orientación CCW obligatoria
        if _area_con_signo(clean) < 0:
            clean = clean[::-1]

        result.append((code, clean))

    return result


def exportar_vem(poligonos_matriz, agg_34, agg_12, agg_38, agg_4,
                 nombre_nodos=archivo_nodos,
                 nombre_conect=archivo_conectividad,
                 ndigits=precision_nodos):
    """
    Genera Input_nodos.txt e Input_conectividad.txt para el programa VEM.

    La malla es conforming: elementos adyacentes comparten exactamente los mismos
    nodos en sus aristas comunes (sin T-junctions).

    Pasos:
      1. Recopilar todos los agregados con su código de material.
      2. Calcular la región de matriz = dominio - unión de todos los agregados.
      3. Filtrar las celdas de matriz (las mismas del gráfico matplotlib).
      4. Reconstruir la malla con polygonize (unary_union de anillos).
      5. Asignar código de material con STRtree.
      6. Mapear nodos, fusionar los muy cercanos y escribir archivos.
    """


    # 1) Juntar todos los agregados con su código de material
    todos_aggs = []
    for code, lst in [(1, agg_34), (2, agg_12), (3, agg_38), (4, agg_4)]:
        if not lst:
            continue
        for p in lst:
            p = reparar(p)
            if p is None or p.is_empty or p.area < area_minima:
                continue
            if isinstance(p, MultiPolygon):
                for g in p.geoms:
                    g = reparar(g)
                    if g and not g.is_empty and g.area >= area_minima:
                        todos_aggs.append((code, g))
            else:
                todos_aggs.append((code, p))

    if not todos_aggs:
        return

    # 2) Unión de todos los agregados (para reconstruir la interfaz agregado-matriz)
    union_aggs = reparar(unary_union([p for _, p in todos_aggs]))

    # 3) Filtrar celdas de matriz (las mismas que se grafican, sin regenerar)
    celdas_mat = []
    for c in poligonos_matriz:
        c = reparar(c)
        if c is None or c.is_empty or c.area < area_minima:
            continue
        if isinstance(c, MultiPolygon):
            for sc in c.geoms:
                sc = reparar(sc)
                if sc and not sc.is_empty and sc.area >= area_minima:
                    celdas_mat.append(sc)
        else:
            # Se incluyen también las celdas que tienen huecos interiores
            # (un agregado dentro), para que su exterior quede en el ring set.
            celdas_mat.append(c)

    # 4) Construir la malla conforming con unary_union + polygonize.
    #    Se usan los bordes de union_aggs (no los individuales) porque tienen
    #    exactamente las mismas coordenadas que los huecos de las celdas de matriz,
    #    ya que ambos vienen del mismo unary_union. Así no quedan micro-slivers
    #    en la interfaz agregado-matriz.
    rings = []
    rings.append(dominio.exterior)
    for c in celdas_mat:
        rings.append(c.exterior)
        for ir in c.interiors:  # anillos interiores si hay un agregado dentro de la celda
            rings.append(ir)
    for g in separar(union_aggs):
        rings.append(g.exterior)

    noded = unary_union(rings)
    raw_cells = list(_polygonize(noded))

    if not raw_cells:
        return

    # 5) Asignar código de material con STRtree (compatible Shapely 1.x y 2.x)
    polys_agg = [p for _, p in todos_aggs]
    codes_agg = [c for c, _ in todos_aggs]
    tree_agg  = STRtree(polys_agg)

    elems_raw = []
    for cell in raw_cells:
        if cell is None or cell.is_empty or cell.area < area_minima:
            continue
        pt = cell.representative_point()
        if not dominio.contains(pt):
            continue
        code  = 5  # código 5 = matriz
        cands = tree_agg.query(pt)
        for cand in cands:
            if not hasattr(cand, 'exterior'):  # Shapely 2.x devuelve índices
                idx = int(cand)
                if polys_agg[idx].contains(pt):
                    code = codes_agg[idx]
                    break
            else:  # Shapely 1.x devuelve geometrías
                for i, ap in enumerate(polys_agg):
                    if ap is cand and ap.contains(pt):
                        code = codes_agg[i]
                        break
                else:
                    continue
                break
        elems_raw.append((code, cell))

    # 6) Mapa de nodos usando floats exactos (sin redondear).
    #    polygonize garantiza que vértices compartidos entre celdas vecinas tienen
    #    exactamente el mismo float de GEOS, así que no hace falta redondear.
    #    Redondear puede generar claves distintas para el mismo vértice si cae
    #    justo en el límite del grid (ej: 12.34505 → 12.3450 vs 12.3451).
    nodo_map = {}
    nodo_xy  = []

    def get_nid(x, y):
        key = (float(x), float(y))
        if key not in nodo_map:
            nodo_map[key] = len(nodo_xy)
            nodo_xy.append(key)
        return nodo_map[key]

    elems = []
    for code, poly in elems_raw:
        raw = list(poly.exterior.coords)
        if len(raw) >= 2 and raw[0] == raw[-1]:
            raw = raw[:-1]
        coords = []
        for x, y in raw:
            pt = (float(x), float(y))
            if not coords or pt != coords[-1]:
                coords.append(pt)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        # Orientación CCW
        area2 = sum((coords[i][0] * coords[(i+1) % len(coords)][1]
                     - coords[(i+1) % len(coords)][0] * coords[i][1])
                    for i in range(len(coords)))
        if area2 < 0:
            coords = coords[::-1]
        ids = [get_nid(x, y) for x, y in coords]
        clean = []
        for nid in ids:
            if not clean or nid != clean[-1]:
                clean.append(nid)
        if len(clean) >= 2 and clean[0] == clean[-1]:
            clean = clean[:-1]
        if len(set(clean)) < 3:
            continue
        elems.append((code, clean))

    # 7) Fusión de nodos muy cercanos con Union-Find.
    #    Captura casos donde dos vértices idénticos físicamente quedan a ambos lados
    #    del límite del grid de snap y reciben IDs distintos. Radio = 10× paso de snap,
    #    mucho menor que el espesor mínimo de matriz (0.12 mm).
    merge_tol = 10.0 ** (2 - ndigits)  # 0.01 mm para ndigits=4
    if len(nodo_xy) > 1:
        _arr = np.array(nodo_xy, dtype=float)
        _kd  = cKDTree(_arr)
        _par = list(range(len(nodo_xy)))

        def _find(x):
            while _par[x] != x:
                _par[x] = _par[_par[x]]
                x = _par[x]
            return x

        for _i in range(len(nodo_xy)):
            for _j in _kd.query_ball_point(_arr[_i], merge_tol):
                if _j != _i:
                    _ri, _rj = _find(_i), _find(_j)
                    if _ri != _rj:
                        _par[max(_ri, _rj)] = min(_ri, _rj)

        _root_map = {}
        _new_xy   = []
        _remap    = [0] * len(nodo_xy)
        for _i in range(len(nodo_xy)):
            _r = _find(_i)
            if _r not in _root_map:
                _root_map[_r] = len(_new_xy)
                _new_xy.append(nodo_xy[_r])
            _remap[_i] = _root_map[_r]

        nodo_xy = _new_xy

        elems2 = []
        for _code, _conn in elems:
            _nc = [_remap[_n] for _n in _conn]
            _cl = []
            for _n in _nc:
                if not _cl or _n != _cl[-1]:
                    _cl.append(_n)
            if len(_cl) >= 2 and _cl[0] == _cl[-1]:
                _cl = _cl[:-1]
            if len(set(_cl)) >= 3:
                elems2.append((_code, _cl))
        elems = elems2

    # 8) Escribir archivos de nodos y conectividad
    N = len(nodo_xy)
    with open(nombre_nodos, "w", encoding="utf-8") as f:
        for i, (xmm, ymm) in enumerate(nodo_xy):
            f.write(f"{i}\t{i + N}\t{xmm / 1000.0:.9f}\t{ymm / 1000.0:.9f}\t0\t0\t0\t0\n")
    with open(nombre_conect, "w", encoding="utf-8") as f:
        for eid, (code, conn) in enumerate(elems):
            f.write("\t".join(map(str, [eid, code, code] + conn)) + "\n")


# 9. Main
def main():
    # Fijar semilla para reproducibilidad (None = aleatoria)
    if semilla is None:
        seed = int(time.time() * 1000) % (2**32 - 1)
    else:
        seed = int(semilla)
    np.random.seed(seed)
    random.seed(seed)

    # 1) Voronoi global sobre el dominio completo
    celdas = voronoi_base()
    indices_libres = list(range(len(celdas)))
    tree = armar_indice(celdas)
    gen_puntos = puntos_grilla(B, H, nx=12, ny=12)
    centros = {1: [], 2: [], 3: [], 4: []}

    # 2) Asignar agregados macro por granulometría, con relajación progresiva
    #    si no se alcanza el área objetivo con los radios iniciales.
    def asignar_con_relajacion(id_clase, area_obj):
        radio_factor = 1.0
        gap_factor = 1.0
        aggs = {}
        usadas_total = set()
        area_acum = 0.0

        while True:
            restante = max(0.0, area_obj - area_acum)
            if restante <= area_obj * 1e-6:
                break

            nuevos, usadas, logrado = asignar_granu(
                celdas, indices_libres, id_clase, restante,
                tree, gen_puntos,
                centros[id_clase], radio_separacion,
                radio_factor=radio_factor, gap_factor=gap_factor
            )

            aggs.update(nuevos)
            usadas_total |= usadas
            area_acum += logrado

            if area_acum > area_obj:
                area_acum = area_obj

            for u in usadas:
                if u in indices_libres:
                    indices_libres.remove(u)

            if area_acum >= area_obj * (1 - tolerancia):
                break
            if len(indices_libres) == 0:
                break

            # Relajar radios para seguir intentando colocar más partículas
            radio_factor *= paso_radio
            gap_factor *= paso_gap

            if radio_factor < min_radio_factor and gap_factor < min_gap_factor:
                break

            radio_factor = max(radio_factor, min_radio_factor)
            gap_factor = max(gap_factor, min_gap_factor)

        return aggs, usadas_total, area_acum

    aggs_34, _, _ = asignar_con_relajacion(1, area_por_clase[0])
    aggs_12, _, _ = asignar_con_relajacion(2, area_por_clase[1])
    aggs_38, _, _ = asignar_con_relajacion(3, area_por_clase[2])

    # 3) Listas de agregados macro
    lista_34 = list(aggs_34.values())
    lista_12 = list(aggs_12.values())
    lista_38 = list(aggs_38.values())

    # 4) Preparar celdas para los agregados #4 (subdividir las muy grandes)
    celdas_4 = preparar_celdas_4(celdas, indices_libres)

    # 5) Asignar agregados #4
    lista_4, _, _ = asignar_fino(celdas_4, area_por_clase[3])

    # 6) Construir la matriz en UNA SOLA llamada sobre dominio - todos los agregados.
    #    Así todas las celdas de matriz vienen del mismo Voronoi y comparten
    #    exactamente los mismos vértices en sus fronteras internas.
    #    Esto es lo que garantiza que el gráfico y los TXT de VEM sean idénticos.
    union_aggs = reparar(unary_union(lista_34 + lista_12 + lista_38 + lista_4))
    region_mat = reparar(dominio.difference(union_aggs)) if union_aggs else dominio
    celdas_mat = subdividir_matriz(region_mat, area_celda_matriz)

    # Exportar TXT para VEM
    if exportar_txts:
        exportar_vem(
            celdas_mat, lista_34, lista_12, lista_38, lista_4,
            nombre_nodos=archivo_nodos, nombre_conect=archivo_conectividad,
            ndigits=precision_nodos
        )

    # Graficar y mostrar resumen
    graficar(celdas_mat, lista_34, lista_12, lista_38, lista_4)
    resumen(lista_34, lista_12, lista_38, lista_4)

if __name__ == "__main__":
    main()
