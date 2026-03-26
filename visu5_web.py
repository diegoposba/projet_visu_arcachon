#!/usr/bin/env python3
"""
Lancement : panel serve visu5_web.py --show

Interactions :
  Slider                      → scrolle la fenêtre (0 → 4800)
  Clic aperçu  (x > WT)       → déplace le centre (a, b)
  Clic bandeau J (bas gauche) → ajuste R0 localement (recalcul partiel)
  Clic vue polaire (haut gch) → rotation angulaire (fast path np.roll)

Dépendances : voir environment.yml
  conda env create -f environment.yml
  conda activate visu_arcachon
"""

import io
import base64
import os
import time
import numpy as np
from PIL import Image
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel as pn
import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Range1d, LabelSet
from bokeh.events import Tap

pn.extension('matplotlib', raw_css=[
    '.bk-Row { gap: 0 !important; margin: 0 !important; padding: 0 !important; }',
    '.bk-panel-models-pane-Matplotlib img { width: 100% !important; height: auto !important; display: block; }',
])

# ── Numba ─────────────────────────────────────────────────────────
# Les fonctions JIT sont dans _numba_kernels.py
# afin que le cache Numba reste valide entre les redémarrages de Panel.
import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_nk = importlib.import_module('_numba_kernels')
_NUMBA_OK        = _nk._NUMBA_OK
_lookup_numba    = _nk._lookup_numba
_lookup_cols_numba = _nk._lookup_cols_numba
_fast_erosion    = _nk._fast_erosion


# ─── Chemins ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(name):
    return os.path.join(SCRIPT_DIR, name)


# ─── Paramètres ────────────────────────────────────────────────────────────────
WT      = 1200
WT_FULL = 6000
HR      = 120
L       = 12

BIVAL_COLORS = {
    1: '#47b947', 2: '#009100', 3: '#007200', 4: '#003E00',
    5: '#FFF177', 6: '#F6EA44', 7: '#FBDE24', 8: '#F4C81A',
    9: '#FFA5A5', 10: '#FF5353', 11: '#FF0000', 12: '#AA0000',
}


def _rgb_to_bokeh(arr_rgb):
    """
    (H, W, 3) uint8  →  (H, W) uint32 RGBA pour Bokeh image_rgba.
    Bokeh rend la ligne 0 en bas → flipud pour que la ligne 0 image
    apparaisse en haut (comme matplotlib origin='upper').
    """
    h, w = arr_rgb.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    rgba  = np.concatenate([arr_rgb, alpha], axis=2)
    img32 = rgba.view(np.uint32).reshape(h, w)
    return np.flipud(img32)


# ══════════════════════════════════════════════════════════════════════════════
class Visu5Engine:
    """
    Moteur de calcul polaire-logarithmique.
    Identique à Visu5 (visu5.py) mais sans matplotlib.
    """

    def __init__(self):
        # ── 1. Image ─────────────────────────────────────────────
        f_img = np.array(Image.open(_p('test4.png')).convert('RGB'))
        self.h, self.w = f_img.shape[:2]
        self.f = f_img

        F = f_img.astype(np.float64)
        for _ in range(3):
            F = (F[0::2] + F[1::2]) / 2
            F = (F[:, 0::2] + F[:, 1::2]) / 2
        self.F  = np.clip(F, 0, 255).astype(np.uint8)
        self.hs = self.h // 8
        self.ws = self.w // 8

        # ── 2. Géoréférencement ──────────────────────────────────
        with open(_p('test4.pgw')) as fh:
            pgw = [float(l.strip()) for l in fh if l.strip()]
        self.pix_x, self.pix_y = pgw[0], pgw[3]
        self.E0,    self.N0    = pgw[4], pgw[5]
        print(f"[img] {self.w}×{self.h} px — aperçu {self.ws}×{self.hs}")

        # ── 3. Couches vecteur ───────────────────────────────────
        self._roads_coords = self._load_layer('data/roads_arca.gpkg', 'routes')
        self._hmax_polys   = self._load_hmax_polys('data/hmax_2023102706_v2.gpkg')

        # ── 4. Filosofi ──────────────────────────────────────────
        filo_path = _p('data/arcachon_filosofi_lamb.gpkg')
        if os.path.exists(filo_path):
            gdf_filo = gpd.read_file(filo_path)
            c = gdf_filo.geometry.centroid
            self._filo_E   = c.x.to_numpy()
            self._filo_N   = c.y.to_numpy()
            self._filo_ind = gdf_filo['Ind'].to_numpy()
            print(f"[filosofi] {len(self._filo_ind)} mailles")
        else:
            self._filo_E = self._filo_N = self._filo_ind = np.empty(0)
            print(f"[filosofi] introuvable : {filo_path}")

        # ── 4b. Communes ──────────────────────────────────────────
        com_path = _p('data/communes.gpkg')
        if os.path.exists(com_path):
            gdf_com = gpd.read_file(com_path)
            if gdf_com.crs is None or gdf_com.crs.to_epsg() != 2154:
                gdf_com = gdf_com.to_crs(epsg=2154)
            centroids = gdf_com.geometry.centroid
            self._commune_E     = centroids.x.to_numpy()
            self._commune_N     = centroids.y.to_numpy()
            self._commune_names = gdf_com['nom_officiel_en_majuscules'].to_numpy()
            print(f"[communes] {len(self._commune_names)} communes")
        else:
            self._commune_E = self._commune_N = np.empty(0)
            self._commune_names = np.empty(0, dtype=object)
            print(f"[communes] introuvable : {com_path}")

        # ── 5. État interactif ───────────────────────────────────
        self.a          = 100
        self.b          = 160
        self.R0         = np.full(WT_FULL, 1200.0)
        self.view_start = 0

        # ── 6. Maillage polaire-log ──────────────────────────────
        r_1d = np.arange(1, self.hs - HR + 1)
        t_1d = 2 * np.pi * np.arange(1, WT_FULL + 1) / WT_FULL
        self.t_mesh, self.r_mesh = np.meshgrid(t_1d, r_1d)
        self.t_mesh = np.ascontiguousarray(self.t_mesh)
        self.r_mesh = np.ascontiguousarray(self.r_mesh)
        self._t0    = float(t_1d[0])

        yr_1d = np.arange(1, HR + 1) - HR / 2
        self.yr_mat = np.flipud(
            np.tile((4 * yr_1d + 1200)[:, None], (1, WT_FULL))
        )

        # ── 7. Cache vecteur ─────────────────────────────────────
        self._roads_proj    = None
        self._hmax_proj     = None
        self._communes_proj = None

        # ── 8. Calcul initial ────────────────────────────────────
        self._recompute()
        self._project_vectors_full()
        self._project_communes_full()

    # ── Chargement couches ────────────────────────────────────────────────────

    def _load_layer(self, rel_path, label):
        full = _p(rel_path)
        if not os.path.exists(full):
            print(f"[{label}] introuvable : {full}")
            return []
        gdf = gpd.read_file(full)
        if gdf.crs is None or gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(epsg=2154)
        coords = self._extract_line_coords(gdf)
        print(f"[{label}] {len(coords)} séquences (EPSG:2154)")
        return coords

    def _load_hmax_polys(self, rel_path):
        full = _p(rel_path)
        if not os.path.exists(full):
            print(f"[hmax_polys] introuvable : {full}")
            return []
        gdf = gpd.read_file(full)
        if gdf.crs is None or gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(epsg=2154)

        def _classify(row):
            q2 = float(row.get('Q2',    0) or 0)
            nb = float(row.get('nb_sc', 0) or 0)
            h_cls = 1 if q2 < 0.50 else (2 if q2 < 1.50 else 3)
            e_cls = 0 if nb <= 9 else (1 if nb <= 17 else (2 if nb <= 24 else 3))
            return (h_cls - 1) * 4 + e_cls + 1

        gdf['_grp'] = gdf.apply(_classify, axis=1)
        dissolved = gdf.dissolve(by='_grp')
        result = []
        for group, row in dissolved.iterrows():
            color = BIVAL_COLORS[group]
            geom  = row.geometry
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            for poly in polys:
                coords = np.array(poly.exterior.coords)[:, :2]
                if len(coords) >= 3:
                    result.append((coords, color))
        print(f"[hmax_polys] {len(result)} anneaux après dissolution")
        return result

    @staticmethod
    def _extract_line_coords(gdf):
        result = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            t = geom.geom_type
            if t == 'LineString':
                arr = np.array(geom.coords)
                if len(arr) >= 2: result.append(arr[:, :2])
            elif t == 'MultiLineString':
                for line in geom.geoms:
                    arr = np.array(line.coords)
                    if len(arr) >= 2: result.append(arr[:, :2])
            elif t == 'Polygon':
                arr = np.array(geom.exterior.coords)
                if len(arr) >= 2: result.append(arr[:, :2])
            elif t == 'MultiPolygon':
                for poly in geom.geoms:
                    arr = np.array(poly.exterior.coords)
                    if len(arr) >= 2: result.append(arr[:, :2])
        return result

    # ── Transformation géo → espace WT_FULL ──────────────────────────────────

    def _geo_to_display(self, xy):
        E, N    = xy[:, 0], xy[:, 1]
        col_px  = (E - self.E0) / self.pix_x
        row_px  = (N - self.N0) / self.pix_y
        dx      = (col_px + 1.0) - 8.0 * self.a
        dy      = (row_px + 1.0) - 8.0 * self.b
        rho     = np.hypot(dx, dy)
        valid   = rho > 1e-6

        t_geo         = np.full(len(E), np.nan)
        t_geo[valid]  = np.arctan2(dy[valid], dx[valid]) % (2 * np.pi)

        total_d   = self.t_mesh[0, 0] - self._t0
        col_float = WT_FULL * (t_geo - total_d) / (2 * np.pi) - 1.0
        col_idx   = np.round(col_float).astype(int) % WT_FULL
        c_val     = self.R0[col_idx]

        r    = np.full(len(E), np.nan)
        mask = valid & (rho > 0)
        r[mask] = self.h / 16.0 * np.log(rho[mask] / 16.0) - c_val[mask]

        row_disp      = (self.hs - HR) - r
        out           = (~valid) | np.isnan(r) | (r < 1) | (r > self.hs - HR)
        cols          = col_idx.astype(float)
        cols[out]     = np.nan
        row_disp[out] = np.nan
        return cols, row_disp

    # ── Cache de projection vecteur ───────────────────────────────────────────

    def _project_vectors_full(self):
        if self._roads_coords:
            lengths  = [len(c) for c in self._roads_coords]
            all_c, all_r = self._geo_to_display(np.vstack(self._roads_coords))
            self._roads_proj = []
            offset = 0
            for l in lengths:
                self._roads_proj.append((all_c[offset:offset+l].copy(),
                                         all_r[offset:offset+l].copy()))
                offset += l
        else:
            self._roads_proj = []

        if self._hmax_polys:
            coords_list = [c for c, _ in self._hmax_polys]
            colors_list = [col for _, col in self._hmax_polys]
            lengths = [len(c) for c in coords_list]
            all_c, all_r = self._geo_to_display(np.vstack(coords_list))
            self._hmax_proj = []
            offset = 0
            for l, color in zip(lengths, colors_list):
                self._hmax_proj.append((all_c[offset:offset+l].copy(),
                                        all_r[offset:offset+l].copy(),
                                        color))
                offset += l
        else:
            self._hmax_proj = []

    def _project_communes_full(self):
        if len(self._commune_E) == 0:
            self._communes_proj = []
            return
        xy = np.column_stack([self._commune_E, self._commune_N])
        cols, rows = self._geo_to_display(xy)
        self._communes_proj = list(zip(cols.tolist(), rows.tolist(),
                                       self._commune_names.tolist()))

    # ── Construction des géométries vecteur ────────────────

    def _build_segments_windowed(self, vs):
        """Retourne liste de (xs, ys) en coordonnées image-mpl (y=0 en haut)."""
        if not self._roads_proj:
            return []
        result = []
        for cols_full, rows in self._roads_proj:
            cols  = cols_full - vs
            valid = (~np.isnan(cols_full) & ~np.isnan(rows)
                     & (cols >= 0) & (cols < WT))
            if valid.sum() < 2:
                continue
            i, n = 0, len(valid)
            while i < n:
                while i < n and not valid[i]: i += 1
                j = i
                while j < n and valid[j]: j += 1
                if j - i >= 2:
                    sc, sr = cols[i:j], rows[i:j]
                    jumps = np.where(np.abs(np.diff(sc)) > WT / 2)[0]
                    if len(jumps) == 0:
                        result.append((sc.tolist(), sr.tolist()))
                    else:
                        prev = 0
                        for jmp in jumps:
                            if jmp + 1 - prev >= 2:
                                result.append((sc[prev:jmp+1].tolist(),
                                               sr[prev:jmp+1].tolist()))
                            prev = jmp + 1
                        if len(sc) - prev >= 2:
                            result.append((sc[prev:].tolist(), sr[prev:].tolist()))
                i = j
        return result

    def _build_polys_windowed(self, vs):
        """Retourne (xs_list, ys_list, colors) en coordonnées image-mpl."""
        if not self._hmax_proj:
            return [], [], []
        xs_list, ys_list, color_list = [], [], []
        for cols_full, rows, color in self._hmax_proj:
            cols  = cols_full - vs
            valid = (~np.isnan(cols_full) & ~np.isnan(rows)
                     & (cols >= 0) & (cols < WT))
            if valid.sum() < 3:
                continue
            sc, sr = cols[valid], rows[valid]
            jumps  = list(np.where(np.abs(np.diff(sc)) > WT / 2)[0] + 1)
            splits = [0] + jumps + [len(sc)]
            for k in range(len(splits) - 1):
                sl = slice(splits[k], splits[k + 1])
                if len(sc[sl]) >= 3:
                    xs_list.append(sc[sl].tolist())
                    ys_list.append(sr[sl].tolist())
                    color_list.append(color)
        return xs_list, ys_list, color_list

    # ── Histogramme ───────────────────────────────────────────────────────────

    def compute_histogram(self, vs):
        """Retourne (centers, sums, n_visible) pour Bokeh."""
        GROUP    = 20
        n_groups = WT // GROUP
        centers  = ((np.arange(n_groups) + 0.5) * GROUP).tolist()
        if len(self._filo_ind) == 0:
            return centers, [0.0] * n_groups, 0

        xy        = np.column_stack([self._filo_E, self._filo_N])
        cols_full, rows = self._geo_to_display(xy)
        cols_win  = cols_full - vs
        visible   = (~np.isnan(cols_full) & ~np.isnan(rows)
                     & (cols_win >= 0) & (cols_win < WT))
        n_vis     = int(visible.sum())
        col_sums  = np.zeros(WT)
        if n_vis > 0:
            col_valid = np.clip(cols_win[visible].astype(int), 0, WT - 1)
            col_sums  = np.bincount(col_valid,
                                    weights=self._filo_ind[visible],
                                    minlength=WT)
        sums = col_sums[:n_groups * GROUP].reshape(n_groups, GROUP).sum(axis=1).tolist()
        return centers, sums, n_vis

    # ── Raster ────────────────────────────────────────────────────────────────

    def _make_overview(self):
        H = self.F.copy()
        a, b = self.a, self.b

        def mark(row, col, half, color):
            r0 = max(0, row - half); r1 = min(H.shape[0], row + half + 1)
            c0 = max(0, col - half); c1 = min(H.shape[1], col + half + 1)
            H[r0:r1, c0:c1] = color

        mark(b, a, 3, [255, 0, 255])
        t_last = float(self.t_mesh[0, -1])
        mark(round(b +  9 * np.sin(t_last)), round(a +  9 * np.cos(t_last)), 2, [255, 0, 255])
        mark(round(b + 17 * np.sin(t_last)), round(a + 17 * np.cos(t_last)), 1, [255, 0, 255])
        return H

    def _make_composite(self, vs):
        G_crop = self.G_full[:, vs:vs + WT]
        J_crop = self.J_full[:, vs:vs + WT]
        left   = np.vstack([np.flipud(G_crop), J_crop])
        right  = self._make_overview()
        right_pad = np.zeros((self.hs, right.shape[1], 3), dtype=np.uint8)
        h_copy = min(self.hs, right.shape[0])
        right_pad[:h_copy] = right[:h_copy]
        return np.hstack([left, right_pad])

    def _lookup(self):
        if _NUMBA_OK:
            return _lookup_numba(
                self.r_mesh, self.t_mesh,
                np.ascontiguousarray(self.R0, dtype=np.float64),
                float(self.a), float(self.b), float(self.h), int(self.w),
            )
        h = self.h
        c = np.tile(self.R0[None, :], (self.hs - HR, 1))
        x = 16 * np.exp(16 * (self.r_mesh + c) / h) * np.cos(self.t_mesh) + 8 * self.a
        y = 16 * np.exp(16 * (self.r_mesh + c) / h) * np.sin(self.t_mesh) + 8 * self.b
        return (np.clip(np.round(y).astype(int), 1, self.h) - 1,
                np.clip(np.round(x).astype(int), 1, self.w) - 1)

    def _lookup_cols(self, cols_idx):
        r_sub  = np.ascontiguousarray(self.r_mesh[:, cols_idx], dtype=np.float64)
        t_sub  = np.ascontiguousarray(self.t_mesh[:, cols_idx], dtype=np.float64)
        R0_sub = np.ascontiguousarray(self.R0[cols_idx],        dtype=np.float64)
        if _NUMBA_OK:
            return _lookup_cols_numba(r_sub, t_sub, R0_sub,
                                      float(self.a), float(self.b),
                                      float(self.h), int(self.w))
        h = self.h
        c = np.tile(R0_sub[None, :], (self.hs - HR, 1))
        x = 16 * np.exp(16 * (r_sub + c) / h) * np.cos(t_sub) + 8 * self.a
        y = 16 * np.exp(16 * (r_sub + c) / h) * np.sin(t_sub) + 8 * self.b
        return (np.clip(np.round(y).astype(int), 1, self.h) - 1,
                np.clip(np.round(x).astype(int), 1, self.w) - 1)

    def _recompute(self, cols_changed=None):
        t0 = time.perf_counter()
        if cols_changed is None or len(cols_changed) == 0:
            self.u_y, self.u_x = self._lookup()
            self.G_full = self.f[self.u_y, self.u_x, :]
        else:
            u_y_sub, u_x_sub = self._lookup_cols(cols_changed)
            self.u_y[:, cols_changed] = u_y_sub
            self.u_x[:, cols_changed] = u_x_sub
            self.G_full[:, cols_changed, :] = self.f[u_y_sub, u_x_sub, :]

        J = np.zeros((HR, WT_FULL, 3), dtype=np.uint8)
        J[:, :, 2] = 128
        raw    = 128.0 * (self.yr_mat < self.R0)
        eroded = _fast_erosion(raw)
        J[:, :, 1] = np.clip(raw - eroded / 2, 0, 255).astype(np.uint8)
        self.J_full = J

        dt   = time.perf_counter() - t0
        mode = 'complet' if cols_changed is None else f'{len(cols_changed)} cols'
        msg  = f"[_recompute] {mode} : {dt * 1000:.1f} ms"
        print(msg)
        return msg

    # ── Dispatch clic ─────────────────────────────────────────────────────────

    def handle_click(self, xg, yg):
        """
        xg, yg : coordonnées dans l'image composite (mpl-like : y=0 en haut).
        Retourne un message de statut.
        """
        if xg > WT:
            # Clic aperçu → déplace le centre
            self.a = int(round(xg - WT))
            self.b = int(round(yg))
            msg = self._recompute()
            self._project_vectors_full()
            self._project_communes_full()
            return f"Aperçu → centre ({self.a}, {self.b}). {msg}"

        elif yg > (self.hs - HR):
            # Clic bandeau J → ajuste R0
            p_col_disp = int(np.clip(round(xg), 1, WT)) - 1
            p_col_full = (self.view_start + p_col_disp) % WT_FULL
            q_row      = int(np.clip(round(yg - (self.hs - HR)), 1, HR)) - 1
            delta      = float(self.yr_mat[q_row, p_col_full] - self.R0[p_col_full])
            base       = np.arange(1, WT_FULL + 1, dtype=float)
            sig        = WT_FULL / L
            global_xg  = self.view_start + xg
            q  = delta * np.exp(-((base              - global_xg) / sig) ** 2)
            q += delta * np.exp(-(((base - WT_FULL)  - global_xg) / sig) ** 2)
            q += delta * np.exp(-(((base + WT_FULL)  - global_xg) / sig) ** 2)
            self.R0 = self.R0 + q
            sig_cols = int(np.ceil(3 * sig))
            affected = np.unique(np.concatenate([
                np.arange(max(0, int(global_xg) - sig_cols),
                          min(WT_FULL, int(global_xg) + sig_cols + 1)),
                np.arange(max(0, int(global_xg - WT_FULL) - sig_cols),
                          min(WT_FULL, int(global_xg - WT_FULL) + sig_cols + 1)),
                np.arange(max(0, int(global_xg + WT_FULL) - sig_cols),
                          min(WT_FULL, int(global_xg + WT_FULL) + sig_cols + 1)),
            ]) % WT_FULL)
            msg = self._recompute(cols_changed=affected)
            # R0 a changé → re-projection complète
            self._project_vectors_full()
            self._project_communes_full()
            return f"Bandeau J. {msg}"

        else:
            # Clic sur la vue polaire → rotation
            global_xg = self.view_start + xg
            d  = 2 * np.pi * global_xg / WT_FULL
            dd = int(max(1, min(WT_FULL, round(WT_FULL * d / (2 * np.pi)))))
            t0 = time.perf_counter()
            self.R0     = np.roll(self.R0,     -dd)
            self.t_mesh = self.t_mesh + d
            self.G_full = np.roll(self.G_full, -dd, axis=1)
            self.J_full = np.roll(self.J_full, -dd, axis=1)
            self.u_x    = np.roll(self.u_x,    -dd, axis=1)
            self.u_y    = np.roll(self.u_y,    -dd, axis=1)
            # Mise à jour rapide du cache vecteur 
            new_roads = []
            for cols_full, rows in self._roads_proj:
                nc = cols_full.copy()
                m  = ~np.isnan(nc)
                nc[m] = (nc[m] - dd) % WT_FULL
                new_roads.append((nc, rows))
            self._roads_proj = new_roads
            new_hmax = []
            for cols_full, rows, color in self._hmax_proj:
                nc = cols_full.copy()
                m  = ~np.isnan(nc)
                nc[m] = (nc[m] - dd) % WT_FULL
                new_hmax.append((nc, rows, color))
            self._hmax_proj = new_hmax
            # Même décalage rapide pour les communes
            self._communes_proj = [
                ((cf - dd) % WT_FULL if not (isinstance(cf, float) and np.isnan(cf)) else cf,
                 r, name)
                for cf, r, name in self._communes_proj
            ]
            dt = (time.perf_counter() - t0) * 1000
            return f"[rotation] roll dd={dd} : {dt:.1f} ms"


# ══════════════════════════════════════════════════════════════════════════════
# Application Panel + Bokeh
# ══════════════════════════════════════════════════════════════════════════════

print("Chargement des données…")
engine = Visu5Engine()
print("Données chargées. Démarrage du serveur Panel…")

# ── Dimensions d'affichage ────────────────────────────────────────────────────
_comp0    = engine._make_composite(0)
IMG_H, IMG_W = _comp0.shape[:2]
HIST_H = 130

# ── Conversion de coordonnées ─────────────────────────────────────────────────

def _mpl_to_bokeh_ys(ys_mpl):
    """Transforme une liste de listes de rangées mpl → coordonnées Bokeh."""
    return [[IMG_H - 1 - r for r in row_list] for row_list in ys_mpl]

def _mpl_to_bokeh_y(r):
    return IMG_H - 1 - r

# ── Sources Bokeh ─────────────────────────────────────────────────────────────

img_source = ColumnDataSource(dict(
    image=[_rgb_to_bokeh(_comp0)],
    x=[0], y=[0], dw=[IMG_W], dh=[IMG_H],
))

# ── Histogramme matplotlib ──────────────────

fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 1.6))
fig_hist.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.12)

def _hist_html(vs):
    gc, gs, nv = engine.compute_histogram(vs)
    ax_hist.cla()
    ax_hist.bar(gc, gs, width=20 * 0.95, color='steelblue', edgecolor='none')
    ax_hist.set_ylabel('Individus', fontsize=8)
    ax_hist.set_title(f'Population Filosofi — {nv} mailles visibles', fontsize=8)
    ax_hist.grid(axis='y', alpha=0.3)
    ax_hist.tick_params(axis='x', labelbottom=False)
    ax_hist.set_xlim(-0.5, WT - 0.5)
    buf = io.BytesIO()
    fig_hist.savefig(buf, format='png', dpi=110, bbox_inches=None)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block">'

hist_pane = pn.pane.HTML(_hist_html(0), sizing_mode='stretch_width')

# Routes
_segs0 = engine._build_segments_windowed(0)
roads_source = ColumnDataSource(dict(
    xs=[s[0] for s in _segs0],
    ys=_mpl_to_bokeh_ys([s[1] for s in _segs0]),
))

# Polygones hmax
_vx0, _vy0, _vc0 = engine._build_polys_windowed(0)
hmax_source = ColumnDataSource(dict(
    xs    =_vx0,
    ys    =_mpl_to_bokeh_ys(_vy0),
    colors=_vc0,
))

# ── Minimap ───────
_ws = engine.ws

def _compute_sector():
    """Secteur angulaire (cône de vue) sur la minimap, en coordonnées Bokeh."""
    vs  = engine.view_start
    theta_start = float(engine.t_mesh[0, vs % WT_FULL])
    theta_end   = float(engine.t_mesh[0, (vs + WT - 1) % WT_FULL])
    cx    = float(WT + engine.a)
    cy_bk = float(IMG_H - 1 - engine.b)   # Bokeh y=0 en bas
    R     = min(_ws, engine.hs) * 0.45
    thetas = np.linspace(theta_start, theta_end, 30)
    # Bokeh y non-flippé → sin est inversé par rapport à mpl
    sx = np.concatenate([[cx], cx + R * np.cos(thetas), [cx]])
    sy = np.concatenate([[cy_bk], cy_bk - R * np.sin(thetas), [cy_bk]])
    return sx.tolist(), sy.tolist()

_sx0, _sy0 = _compute_sector()
sector_source = ColumnDataSource(dict(xs=[_sx0], ys=[_sy0]))

# ── Communes ──────────────────────────────────────────────────────────────────
commune_source = ColumnDataSource(dict(x=[], y=[], text=[]))

def _build_commune_labels(vs):
    """Retourne (xs_bk, ys_bk, texts) des communes visibles dans la fenêtre."""
    xs, ys, texts = [], [], []
    for col_full, row_mpl, name in engine._communes_proj:
        if isinstance(col_full, float) and np.isnan(col_full):
            continue
        if isinstance(row_mpl, float) and np.isnan(row_mpl):
            continue
        x = col_full - vs
        if 5 <= x <= WT - 5:
            xs.append(float(x))
            ys.append(float(_mpl_to_bokeh_y(row_mpl)))
            texts.append(str(name))
    return xs, ys, texts

# ── Points cardinaux ──────────────────────────────────────────────────────────
cardinal_source = ColumnDataSource(dict(x=[], y=[], text=[]))

# ── Figure principale ─────────────────────────────────────────────────────────
x_rng = Range1d(start=0, end=IMG_W, bounds=(0, IMG_W))
y_rng = Range1d(start=0, end=IMG_H, bounds=(0, IMG_H))

p_map = bkp.figure(
    height=450,
    sizing_mode='stretch_width',
    x_range=x_rng, y_range=y_rng,
    tools='tap,wheel_zoom,pan,reset', toolbar_location=None,
    title='Visualisation polaire-logarithmique',
)
p_map.image_rgba(
    image='image', x='x', y='y', dw='dw', dh='dh',
    source=img_source,
)
p_map.patches(
    xs='xs', ys='ys', fill_color='colors',
    source=hmax_source,
    line_color=None, fill_alpha=0.75,
)
p_map.multi_line(
    xs='xs', ys='ys',
    source=roads_source,
    line_color='black', line_width=1.2, line_alpha=0.9,
)
# Secteur de vue sur minimap
p_map.patches(
    xs='xs', ys='ys',
    source=sector_source,
    fill_color='yellow', fill_alpha=0.35,
    line_color='gold', line_width=1.2,
)
# Points cardinaux avec fond semi-transparent
cardinal_labels = LabelSet(
    x='x', y='y', text='text',
    source=cardinal_source,
    text_color='white', text_font_size='14px',
    text_align='center', text_baseline='top',
    text_font_style='bold',
    background_fill_color='rgba(180,0,0,0.85)',
    background_fill_alpha=1.0,
    x_offset=-2, y_offset=-2,
)
# Labels communes
commune_labels = LabelSet(
    x='x', y='y', text='text',
    source=commune_source,
    text_color='white', text_font_size='11px',
    text_align='center', text_baseline='middle',
    text_font_style='bold',
    background_fill_color='rgba(0,0,0,0.60)',
    background_fill_alpha=1.0,
    x_offset=0, y_offset=0,
)
p_map.add_layout(commune_labels)
p_map.add_layout(cardinal_labels)
p_map.axis.visible = False
p_map.grid.visible = False

# ── Légende bivariée ──────────────────────────────────────────────────────────
_H_ROWS  = [('Hmax &lt;0.5 m',   [1, 2, 3, 4]),
             ('Hmax 0.5–1.5 m', [5, 6, 7, 8]),
             ('Hmax &gt;1.5 m', [9, 10, 11, 12])]
_E_COLS  = ['≤9 scén.', '10–17', '18–24', '&gt;24']
_leg_hdr = ('<tr><td style="width:70px"></td>'
            + ''.join(f'<td style="font-size:9px;text-align:center;padding:1px">{c}</td>'
                      for c in _E_COLS) + '</tr>')
_leg_rows = ''.join(
    '<tr><td style="font-size:10px;padding:2px 4px 2px 2px;white-space:nowrap">'
    + label + '</td>'
    + ''.join(f'<td style="background:{BIVAL_COLORS[g]};width:22px;height:16px;'
              f'border:1px solid #555"></td>' for g in grps)
    + '</tr>'
    for label, grps in _H_ROWS)
_legend_html = (
    '<div style="font-family:monospace;padding:6px 4px 4px 4px">'
    '<b style="font-size:11px">Hmax médian × Nb scénarios de submersion</b>'
    f'<table cellspacing="1" style="margin-top:4px;border-collapse:separate">'
    f'{_leg_hdr}{_leg_rows}</table></div>'
)
legend_pane = pn.pane.HTML(_legend_html)

# ── Widgets ───────────────────────────────────────────────────────────────────
slider = pn.widgets.IntSlider(
    name='Vue', start=0, end=WT_FULL - WT, value=0, step=1,
    sizing_mode='stretch_width',
)

status = pn.pane.Str(
    'Prêt — cliquer sur la carte pour interagir.',
    sizing_mode='stretch_width',
    styles={'font-family': 'monospace', 'font-size': '11px', 'color': '#444'},
)


# ── Mise à jour globale ───────────────────────────────────────────────────────

def _push_update(msg):
    """Rafraîchit toutes les sources Bokeh après un changement d'état."""
    vs = engine.view_start

    # Image composite
    img_source.data = dict(
        image=[_rgb_to_bokeh(engine._make_composite(vs))],
        x=[0], y=[0], dw=[IMG_W], dh=[IMG_H],
    )

    # Routes
    segs = engine._build_segments_windowed(vs)
    roads_source.data = dict(
        xs=[s[0] for s in segs],
        ys=_mpl_to_bokeh_ys([s[1] for s in segs]),
    )

    # Polygones hmax
    vx, vy, vc = engine._build_polys_windowed(vs)
    hmax_source.data = dict(
        xs    =vx,
        ys    =_mpl_to_bokeh_ys(vy),
        colors=vc,
    )

    # Histogramme matplotlib
    hist_pane.object = _hist_html(vs)

    # Secteur de vue sur minimap
    sx, sy = _compute_sector()
    sector_source.data = dict(xs=[sx], ys=[sy])

    # Labels communes
    cx, cy, ct = _build_commune_labels(vs)
    commune_source.data = dict(x=cx, y=cy, text=ct)

    # Points cardinaux
    _update_cardinals()

    status.object = msg


def _update_cardinals():
    """Recalcule les positions N/E/S/O dans la fenêtre Bokeh courante."""
    total_d = engine.t_mesh[0, 0] - engine._t0
    vs      = engine.view_start
    xs, ys, texts = [], [], []
    for label, t_geo in [('N', 3 * np.pi / 2), ('E', 0.0),
                         ('S', np.pi / 2), ('O', np.pi)]:
        col_full = int(round(WT_FULL * (t_geo - total_d) / (2 * np.pi) - 1)) % WT_FULL
        x_bk = col_full - vs
        if 10 <= x_bk <= WT - 10:
            xs.append(x_bk)
            ys.append(IMG_H - 25)   # haut de l'image
            texts.append(label)
    cardinal_source.data = dict(x=xs, y=ys, text=texts)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def on_slider(event):
    engine.view_start = event.new
    _push_update(f'Vue : {event.new}')

slider.param.watch(on_slider, 'value')



def on_tap(event):
    """
    Tap Bokeh → dispatch vers engine.handle_click.
    Conversion : Bokeh y (0 en bas) → mpl y (0 en haut).
    L'image a été flipud → bokeh_y = IMG_H - 1 - mpl_y
                         → mpl_y   = IMG_H - 1 - bokeh_y
    """
    bx, by = event.x, event.y
    # Filtrer les clics hors image
    if bx < 0 or bx >= IMG_W or by < 0 or by >= IMG_H:
        return
    xg = bx
    yg = IMG_H - 1 - by
    msg = engine.handle_click(xg, yg)
    _push_update(msg)

p_map.on_event(Tap, on_tap)

# Initialisation communes et points cardinaux
_cx0, _cy0, _ct0 = _build_commune_labels(0)
commune_source.data = dict(x=_cx0, y=_cy0, text=_ct0)
_update_cardinals()

# ── Layout Panel ──────────────────────────────────────────────────────────────

_left_adj   = 0.06
_right_adj  = 0.99
_plot_frac  = _right_adj - _left_adj         
_flex_hist  = round(engine.ws * WT / (IMG_W * _plot_frac - WT) * 0.90)
_flex_right = engine.ws    

top_row = pn.Row(
    pn.Column(hist_pane, sizing_mode='stretch_width',
              styles={'flex': f'{_flex_hist} 1 0%', 'min-width': '0'},
              margin=(0, 0, 0, 0)),
    pn.Column(legend_pane,
              styles={'flex': f'{_flex_right} 1 0%',
                      'padding': '6px 0 0 10px', 'min-width': '0'},
              margin=(0, 0, 0, 0)),
    sizing_mode='stretch_width',
    margin=(0, 0, 0, 0),
)
_flex_total   = _flex_hist + _flex_right
_map_pad_left = f'{_left_adj * _flex_hist / _flex_total * 100:.2f}%'

app = pn.Column(
    top_row,
    pn.Column(
        pn.pane.Bokeh(p_map, sizing_mode='stretch_width'),
        sizing_mode='stretch_width',
        styles={'padding-left': _map_pad_left, 'box-sizing': 'border-box'},
        margin=(0, 0, 0, 0),
    ),
    slider,
    status,
    sizing_mode='stretch_width',
    margin=(5, 5, 5, 5),
)

app.servable()

# ── Point d'entrée direct (sans panel serve) ──────────────────────────────────
if __name__ == '__main__':
    pn.serve(app, show=True, port=5006, title='Submersion — visu5 web')
