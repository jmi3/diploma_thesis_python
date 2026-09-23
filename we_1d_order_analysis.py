"""Řád schématu zvlášť v prostoru a zvlášť v čase.

Míchat obojí v jednom skenu nemá smysl -- když zjemníš dr i dt naráz, změříš
jen ten horší z obou řádů a nepoznáš, který to byl. Proto dva oddělené skeny:

  Sken 1 (prostor): dt drží malé a fixní, dr se půlí  -> čekám 6 (D6)
  Sken 2 (čas):     dr drží fixní, dt se půlí         -> čekám 4 (RK4)

Sken 2 se měří self-konvergencí, ne proti exaktnímu řešení: prostorová chyba je
při fixním dr ve všech třech bězích identická, takže se v rozdílech odečte a
zůstane čistě časová. Proti exaktnímu řešení by ji prostorová chyba přebila.

Pozor na CFL: dt/dr musí zůstat dost malé, jinak explicitní RK spadne.
"""

import numpy as np

from src import verification as v
from src.we_1d_gaussian import exact_psi, run_1d_gaussian

T_MAX = 4.0  # bezpečně pod exit_time = rmax - r0 = 6
GAMMA2 = 0.1
RK_ORDER = 4

# ---------------------------------------------------------------------------
# Sken 1: prostorový řád proti exaktnímu řešení
# ---------------------------------------------------------------------------

DT_FIXED = 0.0025  # tak malé, aby časová chyba RK4 byla zanedbatelná
rows = []
prev = None
for dr in [0.1, 0.05, 0.025]:
    run = run_1d_gaussian(dr, DT_FIXED, T_MAX, output_every=1, gamma2=GAMMA2, rk_order=RK_ORDER)
    err = v.l2_norm(run.psi[-1] - exact_psi(run)[-1], run.dr)
    rows.append(
        {
            "dr": dr,
            "err_l2": err,
            "ratio": prev / err if prev else np.nan,
            "order": np.log2(prev / err) if prev else np.nan,
        }
    )
    prev = err

v.print_table(rows, f"Sken 1 -- prostorový řád (dt = {DT_FIXED}, t = {T_MAX})   čekám 6")

# ---------------------------------------------------------------------------
# Sken 2: časový řád self-konvergencí
# ---------------------------------------------------------------------------

DR_FIXED = 0.05
DT_COARSE = 0.02  # dt/dr = 0.4, CFL v pohodě
psis = []
for k in range(3):
    # output_every se zdvojuje s půlením dt -> snímky padnou na stejné časy
    run = run_1d_gaussian(
        DR_FIXED,
        DT_COARSE / 2**k,
        T_MAX,
        output_every=2**k,
        gamma2=GAMMA2,
        rk_order=RK_ORDER,
    )
    psis.append(run.psi[-1])

# space_factor=1: mřížka se nemění, zjemňuje se jen čas
order, d1, d2 = v.self_convergence(psis[0], psis[1], psis[2], DR_FIXED, space_factor=1)
print(f"\nSken 2 -- časový řád self-konvergencí (dr = {DR_FIXED}, t = {T_MAX})   čekám 4")
print(f"  ||u_dt - u_dt/2||   = {d1:.4e}")
print(f"  ||u_dt/2 - u_dt/4|| = {d2:.4e}")
print(f"  poměr               = {d1 / d2:.2f}   (má být 2^4 = 16)")
print(f"  řád                 = {order:.3f}")
