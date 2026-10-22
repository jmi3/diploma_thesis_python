"""Výstupy pro shrnutí metodiky: obrázky (PDF), tabulky a klíčová čísla (LaTeX fragmenty).

Spuštění z kořene repa:

    python methodology_outputs.py [výstupní adresář]

Výchozí výstupní adresář je ../diploma_thesis_tex/figures. Všechna čísla v textu
shrnutí pocházejí z tohoto skriptu (tabulky tab_*.tex, makra v numbers.tex), nic se
nepřepisuje ručně.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import verification as v
from src.derivatives import deriv_r_o6
from src.parity import Parity
from src.rk_integrator import RKp
from src.we_1d_gaussian import Run, grid, run_1d_gaussian
from src.we_1d_gaussian import exact_psi as exact_psi_1d
from src.we_nd_gaussian import exact_psi as exact_psi_nd
from src.we_nd_gaussian import initial_data as initial_data_nd
from src.we_nd_gaussian import run_gaussian
from src.we_systems import ndim_lin_wave_equation_div_rhs as rhs_div
from src.we_systems import ndim_lin_wave_equation_rhs as rhs_split

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent.parent / "diploma_thesis_tex" / "figures"
)
OUT.mkdir(parents=True, exist_ok=True)

GAMMA2 = 0.1
KO_SIGMAS = [0.0, 0.05, 0.2, 0.5, 1.0]
CFL = 0.25
U0 = 2.1  # střed pulsu procházejícího počátkem
NUMBERS = {}

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Pomocné funkce pro LaTeX
# ---------------------------------------------------------------------------


def sci(x: float) -> str:
    """Číslo ve tvaru $a\\times10^{b}$; inf a nan zvlášť."""
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return r"$\infty$"
    mant, exp = f"{x:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def fix(digits: int):
    return lambda x: "--" if not np.isfinite(x) else f"{x:.{digits}f}"


def text(x) -> str:
    return str(x)


def write_table(name: str, header: list[str], rows: list[list], fmt: list) -> None:
    lines = [
        r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        if row == "midrule":
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(f(x) for f, x in zip(fmt, row, strict=True)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_numbers() -> None:
    lines = [rf"\newcommand{{\num{key}}}{{{value}}}" for key, value in NUMBERS.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Puls procházející počátkem: psi = (f(t - r) - f(t + r)) / r, f(u) = -u exp(-(u - u0)^2)
# ---------------------------------------------------------------------------


def _f(u):
    return -u * np.exp(-np.power(u - U0, 2))


def _df(u):
    return -np.exp(-np.power(u - U0, 2)) * (1 - 2 * u * (u - U0))


def _ddf(u):
    s = u - U0
    return np.exp(-np.power(s, 2)) * (4 * s + 2 * u - 4 * u * np.power(s, 2))


def implosion_psi(t, r):
    r_safe = np.where(r == 0.0, 1.0, r)
    return np.where(r == 0.0, -2.0 * _df(t), (_f(t - r) - _f(t + r)) / r_safe)


def implosion_pi(t, r):
    r_safe = np.where(r == 0.0, 1.0, r)
    return np.where(r == 0.0, 2.0 * _ddf(t), -(_df(t - r) - _df(t + r)) / r_safe)


# ---------------------------------------------------------------------------
# Obecný 3D běh: data "outgoing" (gaussián v r0 = 4) nebo "implosion", dt = CFL * dr
# ---------------------------------------------------------------------------


def run_case(dr, tmax, data, rhs_builder, ko_sigma=0.0, cfl=CFL, rmax=10.0):
    dt = cfl * dr
    r = grid(dr, rmax)
    n = len(r)
    if data == "outgoing":
        psi, phi, pi = initial_data_nd(r, 4.0, 0.5, 3)
    else:
        psi = implosion_psi(0.0, r)
        phi = deriv_r_o6(psi, dr, parity=Parity.Even)
        pi = implosion_pi(0.0, r)
    solver = RKp(order=4)
    solver.Initialize(
        np.concatenate([psi, phi, pi]),
        rhs_builder(
            r_coordinate=r, dimension_n=3, dx=dr, gamma2=GAMMA2, N_of_points=n, ko_sigma=ko_sigma
        ),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        solver.Integrate(t0=0.0, h=dt, tmax=tmax, output_every=max(1, round(0.05 / dt)))
        run = Run.from_solver(
            solver, r=r, meta={"r0": 4.0, "sigma": 0.5, "dimension_n": 3, "dt_integration": dt}
        )
        if data == "outgoing":
            exact, t_exit = exact_psi_nd(run), rmax - 4.0
        else:
            exact, t_exit = implosion_psi(run.t[:, None], run.r[None, :]), rmax + U0
        err = v.l2_norm_history(run.psi - exact, dr)
        con = v.l2_norm_history(v.constraint_violation(run.psi, run.phi, dr), dr)
    return run, err, con, t_exit


def matrix(rhs, n_unknowns):
    """Matice semidiskrétního systému, sloupec j = rhs(e_j) (jako v operator_spectrum)."""
    m = np.zeros((n_unknowns, n_unknowns))
    unit = np.zeros(n_unknowns)
    for j in range(n_unknowns):
        unit[:] = 0.0
        unit[j] = 1.0
        m[:, j] = rhs(0.0, unit)
    return m


def system_matrix(builder, dr, rmax, ko_sigma=0.0):
    r = np.arange(0.0, rmax + dr / 2, dr)
    n = len(r)
    rhs = builder(
        r_coordinate=r, dimension_n=3, dx=dr, gamma2=GAMMA2, N_of_points=n, ko_sigma=ko_sigma
    )
    return matrix(rhs, 3 * n), n


# ---------------------------------------------------------------------------
# 1) Řád prostorové derivace D6 po oblastech
# ---------------------------------------------------------------------------
print("1) test derivace", flush=True)
rows = []
for label, func, dfunc, parity in [
    # Frekvence 4 drží chyby nad strojovou přesností i na nejjemnější mřížce.
    ("even, $\\cos 4r$", lambda r: np.cos(4 * r), lambda r: -4 * np.sin(4 * r), Parity.Even),
    ("odd, $\\sin 4r$", lambda r: np.sin(4 * r), lambda r: 4 * np.cos(4 * r), Parity.Odd),
]:
    res = v.derivative_order_test(
        func, dfunc, [21, 41, 81], r_max=1.0, parity=parity, left_bound_origin=True
    )
    for row in res:
        rows.append(
            [
                label,
                row["dx"],
                row["vnitrek"],
                row["pocatek"],
                row["okraj"],
                row["rad_vnitrek"],
                row["rad_pocatek"],
                row["rad_okraj"],
            ]
        )
        label = ""
write_table(
    "tab_deriv",
    [
        "data",
        r"$\Delta r$",
        "interior",
        "origin",
        "outer",
        "$p_\\mathrm{int}$",
        "$p_\\mathrm{orig}$",
        "$p_\\mathrm{out}$",
    ],
    rows,
    [text, fix(4), sci, sci, sci, fix(2), fix(2), fix(2)],
)

# ---------------------------------------------------------------------------
# 2) Konvergence v prostoru (proti exaktnímu řešení) a v čase (self-konvergence)
# ---------------------------------------------------------------------------
print("2) konvergence", flush=True)
rows = []
for label, runner, exact in [
    (
        "1D",
        lambda dr, dt, oe: run_1d_gaussian(dr, dt, 4.0, rk_order=4, output_every=oe, gamma2=GAMMA2),
        exact_psi_1d,
    ),
    (
        "3D div.",
        lambda dr, dt, oe: run_gaussian(
            dr, dt, 4.0, 3, rhs_div, rk_order=4, output_every=oe, gamma2=GAMMA2
        ),
        exact_psi_nd,
    ),
]:
    errs = []
    for dr in [0.1, 0.05, 0.025]:
        run = runner(dr, 0.0025, 400)
        errs.append(v.l2_norm(run.psi[-1] - exact(run)[-1], run.dr))
    orders = [np.nan] + v.orders_from_errors(errs)
    for dr, e, p in zip([0.1, 0.05, 0.025], errs, orders, strict=True):
        rows.append([label + ", space", dr, "0.0025", e, p])
    psis = [runner(0.05, 0.02 / 2**k, 2**k).psi[-1] for k in range(3)]
    order, d1, d2 = v.self_convergence(*psis, 0.05, space_factor=1)
    rows.append([label + ", time", 0.05, "0.02/0.01/0.005", d2, order])
    rows.append("midrule")
rows.pop()
write_table(
    "tab_conv",
    ["test", r"$\Delta r$", r"$\Delta t$", "error", "order"],
    rows,
    [text, fix(3), text, sci, fix(2)],
)

# ---------------------------------------------------------------------------
# 3) Historie chyby a constraintu, divergenční tvar, oboje data
# ---------------------------------------------------------------------------
print("3) historie chyby", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
for ax, data, tmax, title in [
    (axes[0], "outgoing", 16.0, "outgoing Gaussian"),
    (axes[1], "implosion", 20.0, "pulse through the origin"),
]:
    run, err, con, t_exit = run_case(0.05, tmax, data, rhs_div)
    ax.semilogy(run.t, err, label=r"$\|e\|_2$")
    ax.semilogy(run.t, con, label=r"$\|C\|_2$")
    ax.axvline(t_exit, color="k", ls=":", lw=1, label="pulse exits")
    ax.set_xlabel("$t$")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.5)
axes[0].legend()
fig.savefig(OUT / "fig_history.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Spektrum obou tvarů a škálování max Re(lambda) s dr
# ---------------------------------------------------------------------------
print("4) spektra", flush=True)
fig, ax = plt.subplots(figsize=(6.6, 2.6))
spectra = {}
for builder, label, marker in [(rhs_split, "split", "o"), (rhs_div, "flux", "x")]:
    m, _ = system_matrix(builder, 0.05, 10.0)
    lam = np.linalg.eigvals(m)
    spectra[label] = lam * 0.05
    NUMBERS[f"MinRe{label.capitalize()}"] = f"{lam.real.min() * 0.05:.2f}"
    ax.scatter(lam.real * 0.05, lam.imag * 0.05, s=8, marker=marker, label=label)

# Skupiny vlastních čísel, které nejsou rozebrané (otevřené otázky v textu).
# Hranice pásma D6: maximum modifikovaného vlnového čísla.
_theta = np.linspace(0, np.pi, 200001)
_band_max = ((45 * np.sin(_theta) - 9 * np.sin(2 * _theta) + np.sin(3 * _theta)) / 30).max()
_split = spectra["split"][spectra["split"].imag > 1e-9]
_split_sorted = _split[np.argsort(-_split.real)]
_second = _split_sorted[1]  # druhý největší Re (první je mód u počátku)
_in_band = spectra["split"][np.abs(spectra["split"].imag) <= _band_max]
_bulk = _in_band[(_in_band.real > 0.01) & (_in_band.real < _second.real - 1e-6)]
_flux = spectra["flux"]
_flux_above = np.sort(_flux[_flux.imag > _band_max + 0.005].imag)[::-1]
# Komplexní nízkofrekvenční módy tlumené zřetelně silněji než gamma2 (oba tvary).
_low = np.concatenate(
    [
        s[(s.imag > 1e-3) & (s.imag < 0.1) & (s.real < -0.02) & (s.real > -1.0)]
        for s in spectra.values()
    ]
)
# Čisté poruchy constraintu: lambda = -gamma2 přesně.
_gamma_count = int((np.abs(spectra["split"] + GAMMA2 * 0.05) < 1e-6).sum())
NUMBERS.update(
    {
        "SecondRe": f"{_second.real:.3f}",
        "SecondIm": f"{_second.imag:.3f}",
        "PosInBand": f"{int((_in_band.real > 0.01).sum())}",
        "BulkMax": f"{_bulk.real.max():.3f}",
        "FluxAbove": ", ".join(f"{y:.2f}" for y in _flux_above),
        "FluxMaxRe": f"{_flux.real.max():.4f}",
        "LowReMin": f"{_low.real.min():.3f}",
        "LowReMax": f"{_low.real.max():.3f}",
        "GammaCount": f"{_gamma_count}",
        "GammaDr": f"{-GAMMA2 * 0.05:.3f}",
    }
)
ax.axvline(0, color="k", lw=0.8)
# Mimo rozsah zůstává jedno silně tlumené reálné vlastní číslo (Re ~ -2.6 / dr).
ax.set_xlim(-0.3, 0.3)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda\,\Delta r$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda\,\Delta r$")
ax.set_title(r"spectrum of the semi-discrete operator, $\Delta r = 0.05$, no KO")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
fig.savefig(OUT / "fig_spectrum.pdf")
plt.close(fig)

rows = []
for dr in [0.1, 0.05, 0.025]:
    row = [dr]
    for builder in (rhs_split, rhs_div):
        m, _ = system_matrix(builder, dr, 10.0)
        mx = np.linalg.eigvals(m).real.max()
        row += [mx, mx * dr]
    rows.append(row)
write_table(
    "tab_scaling",
    [
        r"$\Delta r$",
        r"split $\max\mathrm{Re}\,\lambda$",
        r"$\times\Delta r$",
        r"flux $\max\mathrm{Re}\,\lambda$",
        r"$\times\Delta r$",
    ],
    rows,
    [fix(3), fix(4), fix(5), fix(4), fix(5)],
)

# ---------------------------------------------------------------------------
# 5) Závislost na sigma: poruchová teorie, bisekce prahu, struktura módu
# ---------------------------------------------------------------------------
print("5) prah sigma", flush=True)
DR_S = 0.05
m0, n0 = system_matrix(rhs_split, DR_S, 2.0)
q = system_matrix(rhs_split, DR_S, 2.0, ko_sigma=1.0)[0] - m0
lam, vecs = np.linalg.eig(m0)
k = np.argmax(lam.real)
lam0, vec0 = lam[k], vecs[:, k]
mu, left = np.linalg.eig(m0.T)
w = left[:, np.argmin(np.abs(mu - lam0))]
slope = (w @ q @ vec0) / (w @ vec0)
sigma_lin = -lam0.real / slope.real


def max_re(s):
    return np.linalg.eigvals(m0 + s * q).real.max()


lo, hi = 0.5, 1.0
for _ in range(30):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if max_re(mid) > 1e-8 / DR_S else (lo, mid)
sigma_star = hi

theta = np.linspace(0, np.pi, 200001)
band = (45 * np.sin(theta) - 9 * np.sin(2 * theta) + np.sin(3 * theta)) / 30
amp = np.abs(vec0.reshape(3, n0)) ** 2
amp /= amp.sum()

NUMBERS.update(
    {
        "LamRe": f"{lam0.real * DR_S:.5f}",
        "LamIm": f"{abs(lam0.imag) * DR_S:.4f}",
        "SlopeRe": f"{slope.real * DR_S:.4f}",
        "SlopeIm": f"{abs(slope.imag) * DR_S:.4f}",
        "SigmaLin": f"{sigma_lin:.3f}",
        "SigmaStar": f"{sigma_star:.3f}",
        "BandMax": f"{band.max():.3f}",
        "BandTheta": f"{theta[band.argmax()]:.3f}",
        "ModeLocal": f"{100 * amp[:, :6].sum():.1f}",
        "ModePi": f"{100 * amp[2].sum():.0f}",
        "ModePhi": f"{100 * amp[1].sum():.0f}",
    }
)

sig = np.linspace(0, 1, 41)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
axes[0].plot(sig, [max_re(s) * DR_S for s in sig], label=r"$\max\mathrm{Re}\,\lambda\,\Delta r$")
axes[0].plot(
    sig, lam0.real * DR_S + slope.real * DR_S * sig, "--", label="first-order perturbation"
)
axes[0].axvline(sigma_star, color="k", ls=":", lw=1, label=rf"$\sigma^* = {sigma_star:.3f}$")
axes[0].set_xlabel(r"$\sigma$")
axes[0].set_ylim(-0.01, 0.16)
axes[0].legend(fontsize=7)
axes[0].grid(True, ls="--", alpha=0.5)
axes[0].set_title(r"split form, $\Delta r = 0.05$")
for dr, style in [(0.05, "-"), (0.01, "--")]:
    m, n = system_matrix(rhs_split, dr, 2.0)
    lam_d, vec_d = np.linalg.eig(m)
    vd = vec_d[:, np.argmax(lam_d.real)].reshape(3, n)
    scale = np.abs(vd[2]).max()
    j = np.arange(16)
    axes[1].plot(
        j, np.abs(vd[2][:16]) / scale, "o" + style, ms=3, label=rf"$|\pi_j|$, $\Delta r={dr}$"
    )
    axes[1].plot(
        j, np.abs(vd[1][:16]) / scale, "s" + style, ms=3, label=rf"$|\phi_j|$, $\Delta r={dr}$"
    )
axes[1].set_xlabel("grid index $j$  ($r_j = j\\Delta r$)")
axes[1].set_title("unstable eigenvector (normalised)")
axes[1].legend(fontsize=7)
axes[1].grid(True, ls="--", alpha=0.5)
fig.savefig(OUT / "fig_sigma_mode.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) Vliv RK4 na rychlost růstu módu podle CFL
# ---------------------------------------------------------------------------
print("6) RK4", flush=True)
lam_hat = lam0 * DR_S
rows = []
for cfl in [0.1, 0.25, 0.5, 1.0]:
    z = lam_hat * cfl
    amp_rk4 = abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    g = np.log(amp_rk4) / cfl
    rows.append([cfl, g, 100 * (g / lam_hat.real - 1)])
write_table("tab_rk4", ["CFL", r"$g\,\Delta r$", r"change [\%]"], rows, [fix(2), fix(5), fix(2)])

# ---------------------------------------------------------------------------
# 7) KO sken v čase: err po odchodu pulsu, oboje data, tři rozlišení
# ---------------------------------------------------------------------------
print("7) KO sken", flush=True)
results = {}
history = {}
for data, tmax in [("outgoing", 16.0), ("implosion", 20.0)]:
    for dr in [0.05, 0.01, 0.005]:
        for s in KO_SIGMAS:
            run, err, con, t_exit = run_case(dr, tmax, data, rhs_split, ko_sigma=s)
            after = err[run.t > t_exit]
            results[(data, dr, s)] = np.nan if np.isnan(after).all() else np.nanmax(after)
            if data == "implosion" and dr == 0.01:
                history[s] = (run.t, err)
            print(f"   {data} dr={dr} sigma={s}: {results[(data, dr, s)]:.3e}", flush=True)
rows = [
    [f"{s:g}"]
    + [results[(d, dr, s)] for d in ("outgoing", "implosion") for dr in (0.05, 0.01, 0.005)]
    for s in KO_SIGMAS
]
write_table(
    "tab_ko",
    [
        r"$\sigma$",
        "out., 0.05",
        "out., 0.01",
        "out., 0.005",
        "impl., 0.05",
        "impl., 0.01",
        "impl., 0.005",
    ],
    rows,
    [text, sci, sci, sci, sci, sci, sci],
)
fig, ax = plt.subplots(figsize=(6.6, 2.6))
for s, (t, err) in history.items():
    ax.semilogy(t, err, label=rf"$\sigma = {s:g}$")
ax.axvline(10.0 + U0, color="k", ls=":", lw=1)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\|e\|_2$")
ax.set_title(r"split form + KO, pulse through the origin, $\Delta r = 0.01$, CFL $= 0.25$")
ax.legend(fontsize=7, ncol=5)
ax.grid(True, ls="--", alpha=0.5)
fig.savefig(OUT / "fig_ko_history.pdf")
plt.close(fig)

write_numbers()
print("hotovo ->", OUT, flush=True)
