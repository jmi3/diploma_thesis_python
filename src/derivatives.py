import numpy as np


def deriv_r_o4(arr: np.ndarray, dx: float, parity: int = 0, left_bound_origin: bool = True):
    res = np.zeros_like(arr)
    n = len(arr)

    if n < 5:
        raise ValueError("Pole musí mít alespoň 5 bodů pro 4. řád přesnosti.")

    # if we're not dealing with the origin, we set the parity to Undefined
    applied_parity = parity if left_bound_origin else 0

    # Central difference
    res[2:-2] = (arr[:-4] - 8 * arr[1:-3] + 8 * arr[3:-1] - arr[4:]) / (12 * dx)

    # Outer border with 4th order treatment
    res[-1] = (3 * arr[-5] - 16 * arr[-4] + 36 * arr[-3] - 48 * arr[-2] + 25 * arr[-1]) / (12 * dx)
    res[-2] = (-arr[-5] + 6 * arr[-4] - 18 * arr[-3] + 10 * arr[-2] + 3 * arr[-1]) / (12 * dx)

    match applied_parity:
        # Parity.Odd
        case 1:
            # Origin with 4th order treatment for an odd function
            res[0] = (-arr[2] + 8 * arr[1] - (-1) * 8 * arr[1] + (-1) * arr[2]) / (12 * dx)
            # res[1] = (-arr[3] + 8 * arr[2] - 8 * arr[0] + (-1) * arr[1]) / (12 * dx)
            # arr[0] must be zero with Odd parity
            res[1] = (-arr[3] + 8 * arr[2] + (-1) * arr[1]) / (12 * dx)

        # Parity.Even
        case 2:
            # Origin with 4th order treatment for an even function
            # res[0] = (-arr[2] + 8 * arr[1] - 8 * arr[1] + arr[2]) / (12 * dx)
            # res[0] is identically 0 with Even parity
            res[0] = 0
            res[1] = (-arr[3] + 8 * arr[2] - 8 * arr[0] + arr[1]) / (12 * dx)

        # Parity.Undef or not at the origin
        case _:
            res[0] = (-25 * arr[0] + 48 * arr[1] - 36 * arr[2] + 16 * arr[3] - 3 * arr[4]) / (
                12 * dx
            )
            res[1] = (-3 * arr[0] - 10 * arr[1] + 18 * arr[2] - 6 * arr[3] + arr[4]) / (12 * dx)

    return res


def deriv_r_o6(arr: np.ndarray, dx: float, parity: int = 0, left_bound_origin: bool = True):
    res = np.zeros_like(arr)
    n = len(arr)

    if n < 7:
        raise ValueError("Pole musí mít alespoň 7 bodů pro 6. řád přesnosti.")

    # if we're not dealing with the origin, we set the parity to Undefined
    applied_parity = parity if left_bound_origin else 0
    dx60 = 60 * dx
    # Central difference
    res[3:-3] = (
        -arr[:-6] + 9 * arr[1:-5] - 45 * arr[2:-4] + 45 * arr[4:-2] - 9 * arr[5:-1] + arr[6:]
    ) / dx60

    # Outer border: one-sided 6-point stencils, 5th order accurate
    res[-1] = (
        -12 * arr[-6] + 75 * arr[-5] - 200 * arr[-4] + 300 * arr[-3] - 300 * arr[-2] + 137 * arr[-1]
    ) / dx60
    res[-2] = (
        3 * arr[-6] - 20 * arr[-5] + 60 * arr[-4] - 120 * arr[-3] + 65 * arr[-2] + 12 * arr[-1]
    ) / dx60
    res[-3] = (
        -2 * arr[-6] + 15 * arr[-5] - 60 * arr[-4] + 20 * arr[-3] + 30 * arr[-2] - 3 * arr[-1]
    ) / dx60

    # Inner border: with a known parity the ghost points r < 0 are filled by
    # reflection, so the full 6th order central stencil applies; without it we fall
    # back to one-sided stencils, which are only 5th order.
    match applied_parity:
        # Parity.Odd
        case 1:
            # Origin with 6th order treatment for an odd function (arr[-k] = -arr[k])
            res[0] = 2 * (45 * arr[1] - 9 * arr[2] + arr[3]) / dx60
            res[1] = (arr[2] - 9 * arr[1] + 45 * arr[2] - 9 * arr[3] + arr[4]) / dx60
            res[2] = (arr[1] - 45 * arr[1] + 45 * arr[3] - 9 * arr[4] + arr[5]) / dx60

        # Parity.Even
        case 2:
            # Origin with 6th order treatment for an even function (arr[-k] = arr[k])
            res[0] = 0
            res[1] = (-arr[2] + 9 * arr[1] - 45 * arr[0] + 45 * arr[2] - 9 * arr[3] + arr[4]) / dx60
            res[2] = (-arr[1] + 9 * arr[0] - 45 * arr[1] + 45 * arr[3] - 9 * arr[4] + arr[5]) / dx60

        # Parity.Undef or not at the origin
        case _:
            # General 5th order treatment
            res[0] = (
                12 * arr[5]
                - 75 * arr[4]
                + 200 * arr[3]
                - 300 * arr[2]
                + 300 * arr[1]
                - 137 * arr[0]
            ) / dx60
            res[1] = (
                -3 * arr[5] + 20 * arr[4] - 60 * arr[3] + 120 * arr[2] - 65 * arr[1] - 12 * arr[0]
            ) / dx60
            res[2] = (
                2 * arr[5] - 15 * arr[4] + 60 * arr[3] - 20 * arr[2] - 30 * arr[1] + 3 * arr[0]
            ) / dx60

    return res
