import numpy as np

from src.derivatives import deriv_r_o4
from src.parity import Parity


def ndim_lin_wave_equation_rhs(r_coordinate: np.ndarray, dimension_n: int, dx: float, gamma2: float):
    def _system(t: float, y: np.ndarray):
        """
        d/dt[psi] = - pi
        d/dt[phi] = - d/dr[pi] + gamma2 (d/dr[psi] - phi)
        d/dt[pi] = - 1/r^(n-1) d/dr[r^(n-1) phi]
        + Sommerfeldova odchozí podmínka na pravém okraji
        """
        assert len(y) % 3 == 0, "System input is supposed to be made of three equally long functions"

        N = len(y)//3
        
        # 1. Rozbalení vektoru y na jednotlivá pole
        psi = y[0:N]
        phi = y[N : 2 * N]
        pi = y[2 * N : 3 * N]

        # 2. Inicializace derivací (rhs)
        dpsi_dt = np.zeros(N)
        dphi_dt = np.zeros(N)
        dpi_dt = np.zeros(N)

        # --- d/dt[psi] = - pi ---
        dpsi_dt = - pi


        # --- d/dt[phi] = - d/dr[pi] + gamma2 (d/dr[psi] - phi)  - ---
        dphi_dt[1:] = - deriv_r_o4(pi, dx, parity=Parity.Even)[1:] + gamma2 * (deriv_r_o4(psi, dx, parity=Parity.Even) - phi)[1:]
        # Sudost Psi v r zaručuje v počátku Phi (= dPsi/dr) = 0
        dphi_dt[0] = 0.0

        # --- d/dt[pi] = - 1/r^(n-1) d/dr[r^(n-1) phi] ---
        r_to_n_minus_1 =  r_coordinate**(dimension_n-1)
        dpi_dt[1:] = - deriv_r_o4(phi * r_to_n_minus_1, dx, parity=Parity.Odd if N % 2 == 1 else Parity.Even)[1:] / r_to_n_minus_1[1:]
        dpi_dt[0] = - dimension_n * deriv_r_o4(phi, dx, parity=Parity.Odd)[0]

        # --- Sommerfeld (odchozí vlna) ---
        # d_T psi + d_R psi + psi/R = 0, tj. pi = phi + psi/R;
        # derivací podle času: d_T pi = d_T phi + (d_T psi)/R = d_T phi - pi/R
        last_dphi_dt = dphi_dt[-1]
        last_dpi_dt = dpi_dt[-1]
        
        if dimension_n == 1:
            dpi_dt[-1] = last_dphi_dt 
            dphi_dt[-1] = last_dpi_dt
        else:
            factor = pi[-1] * (dimension_n - 1)/(2 * r_coordinate[-1]) 
            dpi_dt[-1] = last_dphi_dt - factor
            dphi_dt[-1] = last_dpi_dt + factor

        # # --- Dirichlet ---
        # dpi_dt[-1] = 0.0
        # dphi_dt[-1] = 0.0

        # 3. Zabalení zpět do jednoho vektoru
        return np.concatenate([dpsi_dt, dphi_dt, dpi_dt])
    return _system

