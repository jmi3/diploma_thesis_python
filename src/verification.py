"""
Test A -- řád derivace:    D6 na funkci se známou derivací. Půlíme dx, chyba má
                           klesnout 2^6 = 64krát.
Test B -- self-konvergence: tři běhy na vnořených mřížkách. Nepotřebuje exaktní
                           řešení, takže funguje i na hyperboloidálních řezech.
Test C -- constraint:      C = D6 psi - phi má klesat jako exp(-gamma2 t).
"""

from itertools import pairwise

import numpy as np

from src.derivatives import deriv_r_o6
from src.parity import Parity


def l2_norm(u: np.ndarray, dx: float) -> float:
    """||u|| = sqrt(dx * sum_j u_j^2) pro jeden časový řez, tvar (N,).

    Na celou historii (M, N) je `l2_norm_history`.

    Faktor dx tam musí být: bez něj norma roste s počtem bodů jako sqrt(N)
    a při zjemňování mřížky by se do všech poměrů zamíchal falešný sqrt(2).
    """
    return float(np.sqrt(dx * np.sum(np.asarray(u) ** 2)))


def l2_norm_history(u: np.ndarray, dx: float) -> np.ndarray:
    """||u(t_i)|| pro každý řez historie: vstup (M, N), výstup (M,)."""
    u = np.asarray(u)
    if u.ndim != 2:
        raise ValueError(f"Čekám historii tvaru (M, N), dostal jsem {u.shape}.")
    return np.sqrt(dx * np.sum(u**2, axis=1))


def orders_from_errors(errors: list[float]) -> list[float]:
    """Z chyb při postupném půlení kroku vrátí řády log2(e_i / e_{i+1}).

    Vrací o jedno číslo méně, než je chyb. Při řádu p mají všechny vyjít p.
    """
    return [float(np.log2(a / b)) for a, b in pairwise(errors)]


def _format_cell(key: str, value) -> str:
    """Text jak je, celá čísla jako celá, řády na 3 desetiny, zbytek vědecky."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{value:d}"
    if key.startswith(("order", "rad_", "ratio")):
        return f"{value:.3f}" if np.isfinite(value) else "--"
    return f"{value:.4e}"


def print_table(rows: list[dict], title: str = "") -> None:
    """Vytiskne seznam slovníků jako tabulku; šířka sloupců podle obsahu."""
    if title:
        print(f"\n{title}")
    if not rows:
        return
    keys = list(rows[0])
    table = [[_format_cell(k, row[k]) for k in keys] for row in rows]
    widths = [max(len(k), *(len(row[i]) for row in table)) for i, k in enumerate(keys)]
    print("  ".join(k.rjust(w) for k, w in zip(keys, widths, strict=True)))
    for row in table:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)))


# ---------------------------------------------------------------------------
# Test A: řád prostorové derivace
# ---------------------------------------------------------------------------


def derivative_order_test(
    func,
    dfunc,
    n_points: list[int],
    r_max: float = 1.0,
    parity: int = 0,
    left_bound_origin: bool = False,
) -> list[dict]:
    """
    Porovná D6 f s analytickou derivací f' na postupně jemnějších mřížkách.

    Args:
        func, dfunc: funkce a její analytická derivace, obě volatelné na numpy poli.
        n_points: počty bodů, každý dvojnásobek předchozího (např. [51, 101, 201]).
        r_max: mřížka je linspace(0, r_max, n).
        parity, left_bound_origin: předá se `deriv_r_o6`. Pro test parity v počátku
            dej parity=Parity.Even (resp. Odd) a left_bound_origin=True.

    Returns:
        Řádky s dx a maximem |chyby| zvlášť ve třech oblastech: `vnitrek`
        (centrální stencil), `pocatek` (3 body u r = 0) a `okraj` (3 body u r_max),
        plus naměřené řády `rad_*`.

    Proč zvlášť: chyba není po mřížce rovnoměrná a maximum přes celou mřížku by
    vždycky změřilo jen ten nejhorší kus. Očekávané řády jsou

        vnitrek  6   (centrální stencil)
        pocatek  6   s paritou (ghost pointy dodá zrcadlení), jinak 5
        okraj    5   (jednostranné stencily, tam parita nepomůže)

    Pozor na strojovou přesnost: pro dx pod ~0.005 padá chyba 6. řádu k 1e-13
    a naměřený řád začne skákat. To není chyba schématu, to je podlaha double.
    """
    rows = []
    prev = {}
    for n in n_points:
        r, dx = np.linspace(0.0, r_max, n, retstep=True)
        err = np.abs(
            deriv_r_o6(func(r), dx, parity=parity, left_bound_origin=left_bound_origin) - dfunc(r)
        )
        now = {
            "vnitrek": float(np.max(err[3:-3])),
            "pocatek": float(np.max(err[:3])),
            "okraj": float(np.max(err[-3:])),
        }
        row = {"n": n, "dx": dx, **now}
        for key, value in now.items():
            old = prev.get(key)
            row[f"rad_{key}"] = np.log2(old / value) if old and value else np.nan
        rows.append(row)
        prev = now
    return rows


# ---------------------------------------------------------------------------
# Test B: self-konvergence
# ---------------------------------------------------------------------------


def self_convergence(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    dx_coarse: float,
    space_factor: int = 2,
) -> tuple[float, float, float]:
    """
    Řád z trojice řešení v jednom čase, každé s 2x menším krokem.

    Args:
        coarse, medium, fine: 1D řezy ve stejném čase, každý s 2x menším krokem.
        dx_coarse: prostorový krok nejhrubšího řezu (jen pro škálování normy).
        space_factor: 2 při zjemňování PROSTORU -- pole mají N, 2N-1, 4N-3 bodů
            a podvzorkují se na nejhrubší mřížku. 1 při zjemňování ČASU, kde
            se mřížka nemění a všechna tři pole mají stejnou délku.

    Returns:
        (řád, ||u_h - u_h/2||, ||u_h/2 - u_h/4||)
    """
    if space_factor < 1:
        raise ValueError("space_factor musí být alespoň 1.")
    m = medium[::space_factor]
    f = fine[:: space_factor**2]
    if not (len(coarse) == len(m) == len(f)):
        raise ValueError(
            f"Nesedí délky: po podvzorkování {len(coarse)}, {len(m)}, {len(f)} bodů. "
            f"Se space_factor={space_factor} čekám vstupy délky N, "
            f"{space_factor}N-{space_factor - 1}, {space_factor**2}N-{space_factor**2 - 1}."
        )
    d1 = l2_norm(coarse - m, dx_coarse)
    d2 = l2_norm(m - f, dx_coarse)
    order = float(np.log2(d1 / d2)) if d2 > 0 else np.nan
    return order, d1, d2


def self_convergence_history(
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    dx_coarse: float,
    space_factor: int = 2,
) -> np.ndarray:
    """
    Totéž jako `self_convergence`, ale pro celou historii -- vrátí řád Q(t).

    Vstupy mají tvar (M, N), (M, 2N-1), (M, 4N-3); časové osy musí být shodné,
    tj. `output_every` musí škálovat s krokem, aby snímky padly na stejné časy.
    V t = 0 je rozdíl nulový, takže Q(0) je nan -- to je v pořádku, počáteční
    data jsou na společných bodech identická.
    """
    m = medium[:, ::space_factor]
    d1 = l2_norm_history(coarse - m, dx_coarse)
    d2 = l2_norm_history(m - fine[:, :: space_factor**2], dx_coarse)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 > 0, np.log2(d1 / d2), np.nan)


# ---------------------------------------------------------------------------
# Test C: porušení constraintu
# ---------------------------------------------------------------------------


def constraint_violation(
    psi: np.ndarray, phi: np.ndarray, dx: float, parity: int = Parity.Even
) -> np.ndarray:
    """
    C = D6 psi - phi, vstup i výstup tvaru (M, N)
    """
    d_psi = np.array([deriv_r_o6(row, dx, parity=parity) for row in np.atleast_2d(psi)])
    return d_psi - np.atleast_2d(phi)


def damping_prediction(t: np.ndarray, c0: float, gamma2: float) -> np.ndarray:
    """||C(0)|| exp(-gamma2 t)."""
    return c0 * np.exp(-gamma2 * np.asarray(t))


# ---------------------------------------------------------------------------
# Spektrum operátoru (jen pro lineární pravé strany)
# ---------------------------------------------------------------------------


def operator_spectrum(rhs, n_unknowns: int) -> np.ndarray:
    """Vlastní čísla matice semidiskrétního systému dy/dt = M y.

    Funguje jen pro LINEÁRNÍ pravou stranu -- matice se sestaví tak, že se `rhs`
    pustí na jednotkové vektory, takže její j-tý sloupec je rhs(e_j).

    K čemu to je: Ljapunovsky je schéma stabilní, jen když max Re(lambda) zůstane
    OMEZENÉ při zjemňování mřížky. Když Re(lambda) roste jako 1/dr, řešení roste
    jako exp(C t / dr) a žádný časový integrátor ani menší dt to nespraví -- je to
    vlastnost prostorové diskretizace. To je nejrychlejší způsob, jak odlišit
    nestabilní diskretizaci od pouhé nepřesnosti.

    Pozor na cenu: `n_unknowns` volání rhs a hustá matice (n_unknowns)^2, takže
    jen na malých mřížkách (řádově stovky bodů).
    """
    matrix = np.zeros((n_unknowns, n_unknowns))
    unit = np.zeros(n_unknowns)
    for j in range(n_unknowns):
        unit[:] = 0.0
        unit[j] = 1.0
        matrix[:, j] = rhs(0.0, unit)
    return np.linalg.eigvals(matrix)
