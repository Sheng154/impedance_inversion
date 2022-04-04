import numpy as np
import matplotlib.pyplot as plt
import bruges


def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


A = bruges.filters.wavelets.sweep(0.2, 0.001, (500, 5000), autocorrelate=False)
plt.plot(A)
plt.show()