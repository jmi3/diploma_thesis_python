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
            t0 (float): Starting time
            h (float): Time step
            dydt (function): function in shape  dy/dt = f(t, y)
            tmax (float, optional): Stopping time of integration. Defaults to None.
            order (int, optional): Order of RK method used. Currently implemented are {1,2,3,4,5}. Defaults to 4.
        """

        self.order = order

        # load RK coefficients
        (c, A, b) = RK_COEFFICIENT_TABLES[order]
        (self.c, self.A, self.b) = (np.array(c), np.array(A), np.array(b))

    # initialization of values before integration process begins
    def Initialize(self, y0: np.ndarray, dydt):
        if len(y0) % 2 == 1:
            raise Exception("Wrong length of initial vector")
        self.y = y0.copy()
        self.ICS = y0.copy()
        self.y_history = [y0.copy()]
        self.dydt = dydt

    # integrate the given system
    def Integrate(self, t0: float, h: float, tmax: float):
        self.t0 = t0
        self.t = t0
        self.h = h

        while True:
            if self.t + h < tmax:
                self.NextStep()
            else:
                if tmax - self.t > 1e-14:
                    self.h = tmax - self.t
                    self.NextStep()
                break

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
    def NextStep(self):
        # obtain k values
        k = self._getKs()
        # update y and t values
        self.y += self.h * sum(self.b[i] * k[i] for i in range(len(self.c)))
        self.t += self.h
        # save y value
        self.y_history.append(self.y.copy())
        return (self.t - self.t0) // self.h

    def GetHistory(self):
        return self.y_history

    @classmethod
    def GetAllImplemented(cls):
        return RK_COEFFICIENT_TABLES.keys()
