import numpy as np

def deriv_r_o4(arr: np.ndarray, dx: float, parity: int = 0, left_bound_origin: bool = True):
    res = np.zeros_like(arr)
    n = len(arr)

    if n < 5:
        raise ValueError("Pole musí mít alespoň 5 bodů pro 4. řád přesnosti.")

    # if we're not dealing with the origin, we set the parity to Undefined
    applied_parity = parity if left_bound_origin else 0

    # Central difference
    res[2:-2] = (arr[:-4] - 8 * arr[1:-3] + 8 * arr[3:-1] - arr[4:] ) / (12 * dx)

    # Outer border with 4th order treatment
    res[-1] = (3*arr[-5] - 16*arr[-4] + 36*arr[-3] - 48*arr[-2] + 25*arr[-1]) / (12 * dx)
    res[-2] = (- arr[-5] + 6*arr[-4] - 18*arr[-3] + 10*arr[-2] + 3*arr[-1]) / (12 * dx)

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
            res[0] = (-25*arr[0] + 48*arr[1] - 36*arr[2] + 16*arr[3] - 3*arr[4]) / (12 * dx)
            res[1] = (-3*arr[0] - 10*arr[1] + 18*arr[2] - 6*arr[3] + arr[4]) / (12 * dx)

    return res