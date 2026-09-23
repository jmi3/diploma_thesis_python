"""Běh 1D lineární vlnové rovnice s gaussovským pulsem na [0, rmax].

Počáteční data pi = phi = d/dr psi dávají čistě pravoběžnou vlnu, ke které existuje
exaktní řešení. Testy správnosti jsou v `src.verification`.
"""

from dataclasses import dataclass, field

import numpy as np

from src.derivatives import deriv_r_o6
from src.parity import Parity
from src.rk_integrator import RKp
from src.we_systems import ndim_lin_wave_equation_div_rhs


@dataclass(frozen=True)
class Run:
    """Odintegrovaný běh rozbalený na jednotlivá pole.

    Integrátor drží stav jako jeden slepený vektor délky 3N; tady je rozřezaný
    na tři pole tvaru (M, N), indexovaná [čas, prostor].

    Attributes:
        r: ekvidistantní mřížka, tvar (N,).
        t: časy uložených snímků, tvar (M,).
        psi, phi, pi: tvar (M, N).
        meta: parametry běhu (r0, sigma, gamma2, dt_integration, ...).
    """

    r: np.ndarray
    t: np.ndarray
    psi: np.ndarray
    phi: np.ndarray
    pi: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def dr(self) -> float:
        return float(self.r[1] - self.r[0])

    @property
    def dt(self) -> float:
        """Rozestup uložených snímků = output_every * integrační krok.

        POZOR: není to integrační krok, ten je v meta["dt_integration"].
        """
        return float(self.t[1] - self.t[0])

    @classmethod
    def from_solver(cls, solver, r: np.ndarray, meta: dict | None = None) -> "Run":
        """Rozbalí y(t) tvaru (M, 3N) na psi, phi, pi."""
        y = np.asarray(solver.GetHistory())
        t = np.asarray(solver.GetTimes())
        n = len(r)
        if y.shape != (len(t), 3 * n):
            raise ValueError(f"Historie má tvar {y.shape}, čekal jsem {(len(t), 3 * n)}.")
        return cls(
            r=r, t=t, psi=y[:, :n], phi=y[:, n : 2 * n], pi=y[:, 2 * n :], meta=dict(meta or {})
        )


# `exact_psi` níž platí jen pro n = 1.
DIMENSION = 1


def grid(dr: float, rmax: float = 10.0) -> np.ndarray:
    """Mřížka r_j = j * dr na [0, rmax].

    Půlení dr dává vnořené mřížky, tj. grid(dr/2)[::2] == grid(dr); na tom stojí
    self-konvergence.
    """
    return np.arange(0.0, rmax + dr, dr)


def initial_data(
    r: np.ndarray, r0: float = 4.0, sigma: float = 0.5, phi_source: str = "discrete"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """psi = exp(-(r - r0)^2 / sigma^2), phi = pi = d/dr psi.

    Volba pi = phi = d/dr psi znamená psi_t = -pi = -psi_r, tedy čistě pravoběžnou vlnu.

    `phi_source` určuje, jak se derivace spočítá, a tím i počáteční hodnotu
    constraintu C = D6 psi - phi:

      * "discrete" -- operátorem D6, tedy stejným jako evoluce: C(0) = 0 na strojovou
        přesnost.
      * "analytic" -- d/dr psi = -2 (r - r0) / sigma^2 * psi, tedy C(0) = -tau, kde tau
        je lokální truncation error D6, čili O(dr^6) ve vnitřku a O(dr^5) na okrajových
        řádcích. Tlumení C(t) = C(0) exp(-gamma2 t)
        se dá testovat jen s C(0) != 0.
    """
    psi = np.exp(-np.power(r - r0, 2) / np.power(sigma, 2))
    if phi_source == "discrete":
        # Parita musí odpovídat evoluci i `constraint_violation`, jinak C(0) != 0.
        d_psi = deriv_r_o6(psi, r[1] - r[0], parity=Parity.Even)
    elif phi_source == "analytic":
        d_psi = -2.0 * (r - r0) / np.power(sigma, 2) * psi
    else:
        raise ValueError(f"Neznámý phi_source: {phi_source!r}")
    return psi, d_psi, d_psi.copy()


def run_1d_gaussian(
    dr: float,
    dt: float,
    tmax: float,
    *,
    rmax: float = 10.0,
    r0: float = 4.0,
    sigma: float = 0.5,
    gamma2: float = 0.1,
    ko_sigma: float = 0.0,
    rk_order: int = 6,
    output_every: int = 1,
    phi_source: str = "discrete",
) -> Run:
    """Odintegruje jeden běh a vrátí ho jako `Run`.

    Args:
        dr, dt: prostorový a časový krok.
        tmax: koncový čas, zaokrouhlí se na nejbližší násobek dt.
        rmax: vnější okraj domény.
        r0, sigma: střed a šířka počátečního gaussiánu.
        gamma2: koeficient tlumení constraintu.
        ko_sigma: síla Kreiss-Oligerovy disipace 8. řádu, 0 = vypnuto.
        rk_order: řád RK; default 6, aby časová chyba nedominovala nad prostorovou
            derivací 6. řádu.
        output_every: ukládá se každý output_every-tý krok. Při porovnávání rozlišení
            ho škáluj s dt, aby všechny běhy měly stejné výstupní časy.
        phi_source: viz `initial_data`.
    """
    r = grid(dr, rmax)
    n_points = len(r)
    psi, phi, pi = initial_data(r, r0, sigma, phi_source=phi_source)

    solver = RKp(order=rk_order)
    solver.Initialize(
        np.concatenate([psi, phi, pi]),
        ndim_lin_wave_equation_div_rhs(
            r_coordinate=r,
            dimension_n=DIMENSION,
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
            "dimension_n": DIMENSION,
            "phi_source": phi_source,
        },
    )


def exact_psi(run: Run) -> np.ndarray:
    """psi(t, r) = exp(-(r - r0 - t)^2 / sigma^2) na mřížce běhu, tvar (M, N).

    Řešení počáteční úlohy na celé přímce. V doméně [0, rmax] platí, dokud puls
    nedorazí na vnější okraj, tj. do t ~ rmax - r0.
    """
    r0, sigma = run.meta["r0"], run.meta["sigma"]
    shifted = run.r[np.newaxis, :] - r0 - run.t[:, np.newaxis]
    return np.exp(-np.power(shifted, 2) / np.power(sigma, 2))


def exit_time(run: Run) -> float:
    """t = rmax - r0, tj. kdy střed pulsu dorazí na vnější okraj."""
    return float(run.r[-1] - run.meta["r0"])
