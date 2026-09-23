"""Kreiss-Oliger artificial dissipation for the radial grid.

The undivided centred difference of order 2p is added to the RHS as

    du/dt += sigma * Q_{2p}[u],    Q_{2p}[u] = (-1)^(p+1) / (2^(2p) * dx) * delta^(2p)[u],

i.e. the operator carries a single power of 1/dx and damps the shortest
wavelength on the grid (the "sawtooth" u_j = (-1)^j) at the rate sigma/dx,
while smooth data are damped only at order dx^(2p-1). The dissipation must
therefore be two orders higher than the scheme it is used with, so that the
formal accuracy is not spoiled:

    ko_order6 (7-point stencil)  ->  4th order scheme,
    ko_order8 (9-point stencil)  ->  6th order scheme (e.g. deriv_r_o6).

Both functions return the operator itself; the caller multiplies by the
dissipation strength sigma (typically 0 <= sigma < 1, larger values make the
explicit time integrator unstable):

    dpi_dt += sigma * ko_order8(pi, dx, parity=Parity.Even)

Near r = 0 the stencil is closed with parity-mirrored ghost points, exactly as
in `derivatives.py`. Where no ghost data are available -- the outer boundary,
and the origin when the parity is unknown -- the dissipation is simply set to
zero instead of being replaced by a one-sided operator.
"""

import numpy as np


def _ghost_points(arr: np.ndarray, half: int, applied_parity: int):
    """Ghost values arr[-half] .. arr[-1] to the left of the origin.

    Odd functions reflect as arr[-k] = -arr[k], even ones as arr[-k] = +arr[k].
    Returns zeros when the parity is unknown; the caller then discards the
    affected points.
    """
    match applied_parity:
        # Parity.Odd
        case 1:
            return -arr[half:0:-1]
        # Parity.Even
        case 2:
            return arr[half:0:-1]
        # Parity.Undef or not at the origin
        case _:
            return np.zeros(half)


def ko_order6(arr: np.ndarray, dx: float, parity: int = 0, left_bound_origin: bool = True):
    """6th order Kreiss-Oliger dissipation, 7-point stencil (for 4th order schemes).

    Q_6[u] = (u[i-3] - 6 u[i-2] + 15 u[i-1] - 20 u[i] + 15 u[i+1] - 6 u[i+2] + u[i+3])
             / (64 dx)
    """
    n = len(arr)

    if n < 7:
        raise ValueError("Pole musí mít alespoň 7 bodů pro 6. řád disipace.")

    # if we're not dealing with the origin, we set the parity to Undefined
    applied_parity = parity if left_bound_origin else 0

    # Ghost points on the left (parity reflection), zeros on the right: the
    # padded array has the same index shift on both sides, so one slicing
    # expression covers the whole grid.
    ext = np.concatenate([_ghost_points(arr, 3, applied_parity), arr, np.zeros(3)])

    res = (
        ext[0:n]
        - 6 * ext[1 : n + 1]
        + 15 * ext[2 : n + 2]
        - 20 * ext[3 : n + 3]
        + 15 * ext[4 : n + 4]
        - 6 * ext[5 : n + 5]
        + ext[6 : n + 6]
    ) / (64 * dx)

    # Outer boundary: no data beyond the grid, no dissipation.
    res[-3:] = 0.0

    # Same at the origin if the parity is not known.
    if applied_parity not in (1, 2):
        res[:3] = 0.0

    return res


def ko_order8(arr: np.ndarray, dx: float, parity: int = 0, left_bound_origin: bool = True):
    """8th order Kreiss-Oliger dissipation, 9-point stencil (for 6th order schemes).

    Q_8[u] = -(u[i-4] - 8 u[i-3] + 28 u[i-2] - 56 u[i-1] + 70 u[i]
               - 56 u[i+1] + 28 u[i+2] - 8 u[i+3] + u[i+4]) / (256 dx)

    The overall minus sign is the (-1)^(p+1) factor with p = 4; without it the
    operator would amplify the sawtooth mode instead of damping it.
    """
    n = len(arr)

    if n < 9:
        raise ValueError("Pole musí mít alespoň 9 bodů pro 8. řád disipace.")

    # if we're not dealing with the origin, we set the parity to Undefined
    applied_parity = parity if left_bound_origin else 0

    ext = np.concatenate([_ghost_points(arr, 4, applied_parity), arr, np.zeros(4)])

    res = (
        -ext[0:n]
        + 8 * ext[1 : n + 1]
        - 28 * ext[2 : n + 2]
        + 56 * ext[3 : n + 3]
        - 70 * ext[4 : n + 4]
        + 56 * ext[5 : n + 5]
        - 28 * ext[6 : n + 6]
        + 8 * ext[7 : n + 7]
        - ext[8 : n + 8]
    ) / (256 * dx)

    # Outer boundary: no data beyond the grid, no dissipation.
    res[-4:] = 0.0

    # Same at the origin if the parity is not known.
    if applied_parity not in (1, 2):
        res[:4] = 0.0

    return res
