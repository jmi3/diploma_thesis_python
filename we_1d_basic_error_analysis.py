"""Jeden běh 1D vlnové rovnice: L2 norma chyby a porušení constraintu v čase."""

import matplotlib.pyplot as plt

from src import verification as v
from src.we_1d_gaussian import exact_psi, exit_time, run_1d_gaussian

# --- Parametry ---
DR = 0.1
DT = 0.005
T_MAX = 30.0
OUTPUT_EVERY = 10
GAMMA2 = 0.1

run = run_1d_gaussian(DR, DT, T_MAX, output_every=OUTPUT_EVERY, gamma2=GAMMA2, rk_order=4)

# --- Chyba vůči exaktnímu řešení ---
# Exaktní řešení je pravoběžný gaussián na celé přímce, takže v naší domény platí
# jen dokud puls nedorazí na vnější okraj -- to je exit_time(run).
error = run.psi - exact_psi(run)
error_l2 = v.l2_norm_history(error, run.dr)

# --- Porušení constraintu C = D6 psi - phi ---
# S phi_source="discrete" (default) je C(0) = 0 identicky, takže všechno naměřené
# vzniklo z okrajových podmínek. Na test tlumení C(t) = C(0) exp(-gamma2 t) je
# potřeba běh s C(0) != 0, tj. phi_source="analytic" -- viz verify.py, test C.
constraint_l2 = v.l2_norm_history(v.constraint_violation(run.psi, run.phi, run.dr), run.dr)

t_exit = exit_time(run)
before = run.t <= t_exit

print(f"exit_time         = {t_exit:.2f}")
print(f"||e||_2 max do    = {error_l2[before].max():.3e}  <- jediné smysluplné číslo")
print(f"||C||_2           = {constraint_l2[0]:.3e} -> {constraint_l2[before][-1]:.3e}")
print(f"||e||_2 max po    = {error_l2[~before].max():.3e}  (odraz od okraje, ne chyba)")
# Za exit_time je exaktní řešení v doméně nulové, takže ||e|| = ||psi|| a měří jen
# to, co po pulsu zbylo -- odraz od Sommerfeldovy podmínky. Konverguje k nule
# s dr^5 (řád vnějšího okraje), takže při dr <= 0.05 je pod zaokrouhlením.

# --- Grafy ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.semilogy(run.t, error_l2, lw=1.5, label=r"$\|e\|_2$")
ax1.semilogy(run.t, constraint_l2, lw=1.5, label=r"$\|C\|_2$")
ax1.axvline(exit_time(run), color="black", lw=1, ls=":", label="odchod pulsu")
ax1.set_xlabel("t")
ax1.set_title(f"L2 normy, dr = {run.dr:g}")
ax1.grid(True, ls="--", alpha=0.6)
ax1.legend()

scale = abs(error).max() or 1.0
mesh = ax2.pcolormesh(
    run.r, run.t, error, cmap="RdBu_r", vmin=-scale, vmax=scale, shading="nearest"
)
fig.colorbar(mesh, ax=ax2, label="e")
ax2.set_xlabel("r")
ax2.set_ylabel("t")
ax2.set_title(f"e(t, r), dr = {run.dr:g}")

fig.tight_layout()
plt.show()
