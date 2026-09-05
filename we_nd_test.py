import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

from src.derivatives import deriv_r_o4
from src.we_systems import ndim_lin_wave_equation_rhs
from src.rk_integrator import RKp, RK_COEFFICIENT_TABLES

# --- Parametry gridu ---
dr = 0.01
r_min, r_max = 0, 10.0
r = np.arange(r_min, r_max + dr, dr)
N = len(r)

# --- Parametry pulsu ---
r0 = 4.0  # Střed pulsu
sigma = 0.5  # Šířka pulsu

# --- Parametry integrace ---
tmin = 0.0
timestep = 0.005
tmax = 100.0


# --- Výpočet počátečních polí ---
# f = exp(-(r-r0)^2 / sigma^2)
psi = np.exp(-np.power(r - r0, 2) / np.power(sigma, 2))

# pi inicializujeme jako derivaci f
phi = deriv_r_o4(psi, dr)

# xi inicializujeme jako 0 -- odpovídá statickému startu
pi = np.zeros(N)

## Můžeme mít i odcházející vlnu
# pi = -deriv_r_o4(f, dr)

# --- Zabalení pro třídu RKp ---
y0 = np.concatenate([psi, phi, pi])

if len(y0) % 2 != 0:
    # Pokud je délka lichá, upravíme grid o jeden bod, aby Initialize nehodilo chybu
    r = r[:-1]
    psi = psi[:-1]
    phi = phi[:-1]
    pi = pi[:-1]
    y0 = np.concatenate([psi, phi, pi])
    N = len(r)

print(f"Počáteční data připravena. Celková délka vektoru y0: {len(y0)} (3x {N})")

# Máme 4. řád derivace, proto raději 6. řád RK
solver = RKp(order=6)
solver.Initialize(y0, ndim_lin_wave_equation_rhs(r, 3, dr, 0.1))
solver.Integrate(t0=tmin, h=timestep, tmax=tmax)

print(f"Integrace ukončena")

# Převedení historie na numpy array [časové_kroky, prostorové_body]
history_full = np.array(solver.GetHistory())

history = history_full[:, 0:N]  # extrahujeme pouze psi (prvních N prvků)



# --- ANIMACE ---
every_nth_frame = 10

fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.plot(r, history[0, :], lw=2, color="firebrick")

ax.set_xlim(r_min, r_max)
ax.set_ylim(-1, 1)
ax.set_title("Animace výsledků")
ax.set_xlabel("Prostor (x)")
ax.set_ylabel("Amplituda (u)")
ax.grid(True, linestyle="--", alpha=0.6)


def update(frame):
    line.set_ydata(history[frame * every_nth_frame, :])
    return (line,)


ani = FuncAnimation(fig, update, frames=len(history) // every_nth_frame, interval=20*(every_nth_frame//10), blit=True, cache_frame_data=False)

plt.show()