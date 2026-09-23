import numpy as np

RK_COEFFICIENT_TABLES = {
    1: ([0], [[0]], [1]),
    2: ([0, 1 / 2], [[0, 0], [1 / 2, 0]], [0, 1]),
    3: ([0, 1 / 2, 1], [[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]], [1 / 6, 2 / 3, 1 / 6]),
    4: (
        [0, 1 / 2, 1 / 2, 1],
        [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    ),
    5: (
        [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ),
    6: (
        [0, 1 / 3, 2 / 3, 1 / 3, 1 / 2, 1 / 2, 1],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 3, 0, 0, 0, 0, 0, 0],
            [0, 2 / 3, 0, 0, 0, 0, 0],
            [1 / 12, 1 / 3, -1 / 12, 0, 0, 0, 0],
            [-1 / 16, 9 / 8, -3 / 16, -3 / 8, 0, 0, 0],
            [0, 9 / 8, -3 / 8, -3 / 4, 1 / 2, 0, 0],
            [9 / 44, -9 / 11, 63 / 44, 18 / 11, 0, -16 / 11, 0],
        ],
        [11 / 120, 0, 27 / 40, 27 / 40, -4 / 15, -4 / 15, 11 / 120],
    ),
}


class RKp:
    def __init__(self, order: int = 4):
        """
        Args:
            order (int, optional): Řád použité RK metody. Implementované jsou klíče
                RK_COEFFICIENT_TABLES, tj. {1,2,3,4,5,6}. Defaults to 4.
        """

        self.order = order

        # load RK coefficients
        (c, A, b) = RK_COEFFICIENT_TABLES[order]
        (self.c, self.A, self.b) = (np.array(c), np.array(A), np.array(b))

    # initialization of values before integration process begins
    def Initialize(self, y0: np.ndarray, dydt):
        """
        Args:
            y0 (np.ndarray): Počáteční data. Integrátor jejich vnitřní strukturu neřeší
                (kolik polí je slepené za sebou je věc `dydt`, ne integrátoru).
            dydt (function): function in shape  dy/dt = f(t, y)
        """
        self.y = y0.copy()
        self.ICS = y0.copy()
        self.dydt = dydt
        self.t0 = 0.0
        self.t = 0.0
        self.h = 0.0
        self.step_index = 0
        self.y_history = [y0.copy()]
        self.t_history = [0.0]

    # integrate the given system
    def Integrate(self, t0: float, h: float, tmax: float, output_every: int = 1):
        """
        Odintegruje přesně `round((tmax - t0) / h)` kroků o pevné délce `h`. Časová osa
        je díky tomu ekvidistantní (`t_i = t0 + i * h`) a `tmax` se zaokrouhlí na
        nejbližší násobek `h`. Volání resetuje běh na počáteční data z `Initialize`.

        Args:
            t0 (float): Starting time
            h (float): Time step
            tmax (float): Stopping time of integration
            output_every (int, optional): Do historie se zapisuje každý `output_every`-tý
                krok; počáteční a koncový stav vždy. Defaults to 1.
        """
        if h <= 0:
            raise ValueError("Krok h musí být kladný.")
        if output_every < 1:
            raise ValueError("output_every musí být alespoň 1.")

        self.t0 = t0
        self.t = t0
        self.h = h
        self.step_index = 0
        self.y = self.ICS.copy()
        self.y_history = [self.ICS.copy()]
        self.t_history = [t0]

        n_steps = round((tmax - t0) / h)
        for i in range(1, n_steps + 1):
            self.NextStep(store=(i % output_every == 0 or i == n_steps))

    # calculate ks for general Butcher table
    def _getKs(self):
        k = np.zeros((len(self.c),) + np.shape(self.y))
        for i in range(len(self.c)):
            t = self.t + self.h * self.c[i]

            dy = np.zeros_like(self.y)
            for j in range(i):
                dy += self.A[i, j] * k[j]
            y = self.y + self.h * dy

            k[i] = self.dydt(t, y)
        return k.copy()

    # perform one step of RKp method
    def NextStep(self, store: bool = True):
        # obtain k values
        k = self._getKs()
        # update y and t values
        self.y += self.h * sum(self.b[i] * k[i] for i in range(len(self.c)))
        self.step_index += 1
        # čas z indexu kroku, ne akumulací t += h -- ta po desetitisících kroků driftuje
        self.t = self.t0 + self.step_index * self.h
        # save y value
        if store:
            self.y_history.append(self.y.copy())
            self.t_history.append(self.t)
        return self.step_index

    def GetHistory(self):
        return self.y_history

    def GetTimes(self):
        """Časy odpovídající řádkům `GetHistory()`."""
        return self.t_history

    @classmethod
    def GetAllImplemented(cls):
        return RK_COEFFICIENT_TABLES.keys()
