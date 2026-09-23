"""Běh n-rozměrné vlnové rovnice s odchozím gaussovským pulsem.

Slouží k porovnání divergenčního a nedivergenčního tvaru rovnice pro pi:

    divergenční:     d/dt[pi] = -1/r^(n-1) d/dr[r^(n-1) phi]
    nedivergenční:   d/dt[pi] = -d/dr[phi] - (n-1)/r phi

Analyticky je to totéž, diskrétně ne. Pro n = 1 jsou ale oba tvary i diskrétně
totožné (člen (n-1)/r zmizí a r^(n-1) = 1), takže porovnání má smysl až od n = 2.
Fyzikálně zajímavý je n = 3, sférická symetrie.

Na rozdíl od `we_1d_gaussian` jsou počáteční data odchozí i v n rozměrech, tj.
pi = phi + (n-1)/2 * psi/r, ne pi = phi.
"""

import numpy as np

from src.derivatives import deriv_r_o6
from src.parity import Parity
from src.rk_integrator import RKp
from src.we_1d_gaussian import Run, exit_time, grid

# `grid` a `exit_time` jsou stejné jako v 1D případě, jen se reexportují, ať je
# volající nemusí tahat ze dvou modulů.
__all__ = ["Run", "exact_psi", "exit_time", "grid", "initial_data", "run_gaussian"]


def initial_data(
    r: np.ndarray, r0: float, sigma: float, dimension_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Odchozí gaussovský puls: psi = exp(-((r-r0)/sigma)^2), pi = phi + a*psi/r.

    Odchozí řešení v n rozměrech má tvar psi = U(r-t) / r^a s a = (n-1)/2. Z toho

        pi  = -d/dt psi = U'(r-t) / r^a,
        phi =  d/dr psi = U'(r-t) / r^a - a U(r-t) / r^(a+1) = pi - a psi/r,

    tedy pi = phi + a psi/r. To je zároveň Sommerfeldova podmínka, kterou používá
    `we_systems` na vnějším okraji.

    phi se počítá diskrétním operátorem (ne analyticky), takže počáteční porušení
    constraintu C = D6 psi - phi je nulové na strojovou přesnost.
    """
    psi = np.exp(-np.power((r - r0) / sigma, 2))
    phi = deriv_r_o6(psi, r[1] - r[0], parity=Parity.Even)

    a = (dimension_n - 1) / 2
    pi = phi.copy()
    if a:
        # V r = 0 je psi(0) ~ exp(-(r0/sigma)^2), pro r0=4, sigma=0.5 tedy e^-64.
        # Limita a*psi/r je tam fakticky nula, takže pi[0] = phi[0].
        pi[1:] = phi[1:] + a * psi[1:] / r[1:]
    return psi, phi, pi


def run_gaussian(
    dr: float,
    dt: float,
    tmax: float,
    dimension_n: int,
    rhs_builder,
    rmax: float = 10.0,
    r0: float = 4.0,
    sigma: float = 0.5,
    gamma2: float = 0.1,
    ko_sigma: float = 0.0,
    rk_order: int = 4,
    output_every: int = 1,
) -> Run:
    """Odintegruje jeden běh a vrátí ho jako `Run`.

    Args:
        dr, dt: prostorový a časový krok.
        tmax: koncový čas, zaokrouhlí se na nejbližší násobek dt.
        dimension_n: počet prostorových rozměrů (3 = sférická symetrie).
        rhs_builder: `ndim_lin_wave_equation_rhs` (nedivergenční) nebo
            `ndim_lin_wave_equation_div_rhs` (divergenční) z `we_systems`.
        ko_sigma: síla Kreiss-Oligerovy disipace 8. řádu, 0 = vypnuto.
        output_every: ukládá se každý output_every-tý krok. Při porovnávání
            rozlišení ho škáluj s dt, aby běhy měly stejné výstupní časy.
    """
    r = grid(dr, rmax)
    n_points = len(r)
    psi, phi, pi = initial_data(r, r0, sigma, dimension_n)

    solver = RKp(order=rk_order)
    solver.Initialize(
        np.concatenate([psi, phi, pi]),
        rhs_builder(
            r_coordinate=r,
            dimension_n=dimension_n,
            dx=dr,
            gamma2=gamma2,
            N_of_points=n_points,
            ko_sigma=ko_sigma,
        ),
    )
    solver.Integrate(t0=0.0, h=dt, tmax=tmax, output_every=output_every)

    return Run.from_solver(
        solver,
        r=r,
        meta={
            "r0": r0,
            "sigma": sigma,
            "gamma2": gamma2,
            "ko_sigma": ko_sigma,
            "rk_order": rk_order,
            "dt_integration": dt,
            "dimension_n": dimension_n,
            "system": rhs_builder.__name__,
        },
    )


def exact_psi(run: Run) -> np.ndarray:
    """Odchozí řešení psi(t, r) = U(r - t) / r^a, a = (n-1)/2, tvar (M, N).

    U(x) = x^a exp(-((x - r0)/sigma)^2), takže v t = 0 vyjde psi = exp(-((r-r0)/sigma)^2)
    pro každé n.

    Platí jen pro n = 1 a n = 3 -- jen tam má radiální vlnová rovnice d'Alembertovo
    řešení tohoto tvaru (pro sudé n Huygensův princip neplatí).

    Dvě omezení, stejná jako v 1D:
      * platí, dokud puls nedorazí na vnější okraj, tj. do t = rmax - r0 (`exit_time`),
      * předpokládá, že levý chvost je pod strojovou přesností, aby řešení
        neodporovalo regularitě v počátku. Pro r0=4, sigma=0.5 je exp(-64) ~ 1e-28.
    """
    n = run.meta["dimension_n"]
    if n not in (1, 3):
        raise ValueError(f"Exaktní řešení mám jen pro n = 1 a n = 3, dostal jsem n = {n}.")

    r0, sigma = run.meta["r0"], run.meta["sigma"]
    x = run.r[np.newaxis, :] - run.t[:, np.newaxis]
    gauss = np.exp(-np.power((x - r0) / sigma, 2))
    if n == 1:
        return gauss

    # n = 3: psi = (r - t) * gauss / r. V r = 0 je limita nulová -- puls tam nedojde.
    out = np.zeros_like(gauss)
    out[:, 1:] = x[:, 1:] * gauss[:, 1:] / run.r[np.newaxis, 1:]
    return out
