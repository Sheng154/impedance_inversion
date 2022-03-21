import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import random
import segyio

# model
tmax = 0.2
tmin = 0
xt = np.arange(0, 200)
# impedance range
max_guess, min_guess = 2800, 1600
max_imp, min_imp = 3000, 1500
# wavelet parameters
f = 50
length = 0.1
dt = 0.001  # 1ms, 1,000Hz
# Input seismogram
lenSeis = 200 + 1
# Optimisation criterion
L1 = True
###########################

nsamp = int((tmax - tmin) / dt) + 1
t = []
for i in range(0, nsamp):
    t.append(i * dt)

np.random.seed(12)

# Layer1 = np.ones(40) * np.random.normal((2200, 300))
Layer1 = np.ones(40) * (np.random.random()*(max_guess-min_guess)) + min_guess
Layer2 = np.ones(40) * (np.random.random()*(max_guess-min_guess)) + min_guess
Layer3 = np.ones(40) * (np.random.random()*(max_guess-min_guess)) + min_guess
Layer4 = np.ones(40) * (np.random.random()*(max_guess-min_guess)) + min_guess
Layer5 = np.ones(41) * (np.random.random()*(max_guess-min_guess)) + min_guess

# Layer2 = np.ones(40) * np.random.normal((2200, 300))
#Layer3 = np.ones(40) * np.random.normal((2200, 300))
#Layer4 = np.ones(40) * np.random.normal((2200, 300))
#Layer5 = np.ones(40) * np.random.normal((2200, 300))
imp = np.concatenate((Layer1, Layer2, Layer3, Layer4, Layer5))

def calcuRc(imp):
    Rc = []
    nsamp = np.shape(imp)[0]
    for i in range(0, nsamp - 1):
        Rc.append((imp[i + 1] - imp[i]) / (imp[i + 1] + imp[i]))
    return Rc


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


wavelet = ricker(f, length, dt)
Rc = calcuRc(imp)

for i in range(np.shape(Rc)[0]):
    if abs(Rc[i]) < 0.005:
        Rc[i] = 0


def SyntheticTrace(Rc, wavelet, nsamp):
    noise = np.random.normal(0, 0.3, nsamp-1)
    noise_smoothed = ndi.uniform_filter1d(noise, size=7)  # Controls the noise level
    synthetic_trace = np.convolve(Rc, wavelet, mode='same')
    synthetic_norm = synthetic_trace / max(synthetic_trace)
    synthetic_ctm = synthetic_norm + noise_smoothed
    return synthetic_trace, synthetic_norm, synthetic_ctm


Synthetic_raw, synthetic, synthetic_contaminated = SyntheticTrace(Rc, wavelet, nsamp)


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1 / (2 * dt), len(power_spectrum))
    return frequency, power_spectrum


f_trace, p_trace = power(synthetic_contaminated)
f_imp, p_imp = power(imp)
plt.plot(f_trace, p_trace, 'r')
plt.show()
plt.plot(f_imp, p_imp, 'b')
plt.show()
#plt.plot(synthetic_contaminated, 'b')
#plt.show()



