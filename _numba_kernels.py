"""
Fonctions Numba JIT isolées dans un module au nom stable.
Importé par visu5.py et visu5_web.py — le cache Numba est lié à CE fichier,
pas au module temporaire créé par Panel/Bokeh au démarrage du serveur.
"""
import numpy as np

_NUMBA_OK = False

try:
    import numba

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _lookup_numba(r_mesh, t_mesh, R0, a, b, h, w):
        nrows, ncols = r_mesh.shape
        u_y = np.empty((nrows, ncols), dtype=np.int32)
        u_x = np.empty((nrows, ncols), dtype=np.int32)
        for i in numba.prange(nrows):
            for j in range(ncols):
                c       = R0[j]
                exp_val = 16.0 * np.exp(16.0 * (r_mesh[i, j] + c) / h)
                x = exp_val * np.cos(t_mesh[i, j]) + 8.0 * a
                y = exp_val * np.sin(t_mesh[i, j]) + 8.0 * b
                u_x[i, j] = max(0, min(int(round(x)), w - 1))
                u_y[i, j] = max(0, min(int(round(y)), h - 1))
        return u_y, u_x

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _lookup_cols_numba(r_sub, t_sub, R0_sub, a, b, h, w):
        nrows, ncols = r_sub.shape
        u_y = np.empty((nrows, ncols), dtype=np.int32)
        u_x = np.empty((nrows, ncols), dtype=np.int32)
        for i in numba.prange(nrows):
            for j in range(ncols):
                c       = R0_sub[j]
                exp_val = 16.0 * np.exp(16.0 * (r_sub[i, j] + c) / h)
                x = exp_val * np.cos(t_sub[i, j]) + 8.0 * a
                y = exp_val * np.sin(t_sub[i, j]) + 8.0 * b
                u_x[i, j] = max(0, min(int(round(x)), w - 1))
                u_y[i, j] = max(0, min(int(round(y)), h - 1))
        return u_y, u_x

    _NUMBA_OK = True
    print("[numba] JIT activé — première exécution lente (~5 s, puis cache).")

except ImportError:
    print("[numba] Non installé (pip install numba pour accélérer).")

    def _lookup_numba(*args, **kwargs):
        raise RuntimeError("numba non disponible")

    def _lookup_cols_numba(*args, **kwargs):
        raise RuntimeError("numba non disponible")


def _fast_erosion(arr):
    """Érosion 3×3 minimum, pure numpy."""
    pad = np.pad(arr, 1, mode='edge')
    return np.minimum.reduce([
        pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:],
        pad[1:-1, :-2], arr,           pad[1:-1, 2:],
        pad[2:,  :-2],  pad[2:, 1:-1], pad[2:,  2:],
    ])
