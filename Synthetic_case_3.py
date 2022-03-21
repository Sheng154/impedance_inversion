import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import random
import segyio
np.random.seed(1234)
'''
# model
tmax = 0.2
tmin = 0
xt = np.arange(0, 200)
# impedance range
max_guess, min_guess = 4500, 1000
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
t = np.linspace(tmin, tmax, nsamp)
np.random.seed(12)


def createModel(t):
    imp_LF = 2250 + 200 * np.sin(2*np.pi*20 * t + 10) + 100 * np.sin(2*np.pi*80 * t - 2)  # Determines the range of synthetic impedance model
    return imp_LF
imp_syn = createModel(t)
plt.plot(imp_syn)
plt.show()
def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1/(2*dt), len(power_spectrum))
    return frequency, power_spectrum


def calcuRc(imp):
    Rc = []
    #nsamp = np.shape(imp)[0]
    for i in range(0, nsamp - 1):
        Rc.append((imp[i + 1] - imp[i]) / (imp[i + 1] + imp[i]))
    return Rc


rc = calcuRc(imp_syn)


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


wavelet = ricker(f, length, dt)
#imp = createModel(nsamp)
#print(imp)
#Rc = calcuRc(imp)

for sample in range(np.shape(rc)[0]):
    if abs(rc[sample]) < 0.005:
        rc[sample] = 0


def SyntheticTrace(Rc, wavelet, nsamp):
    noise = np.random.normal(0, 0.3, nsamp-1)
    noise_smoothed = ndi.uniform_filter1d(noise, size=7)  # Controls the noise level
    synthetic_trace = np.convolve(Rc, wavelet, mode='same')
    synthetic_norm = synthetic_trace / max(synthetic_trace)
    synthetic_ctm = synthetic_norm + noise_smoothed
    return synthetic_trace, synthetic_norm, synthetic_ctm


Synthetic_raw, synthetic, synthetic_contaminated = SyntheticTrace(rc, wavelet, nsamp)


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1/(2*dt), len(power_spectrum))
    return frequency, power_spectrum


f_rc, p_rc = power(rc)
f_imp, p_imp = power(imp_syn)
plt.plot(f_imp, p_imp, 'r')
plt.ylabel('power of imp')
plt.show()
plt.plot(f_rc, p_rc, 'b')
plt.ylabel('power of rc')
plt.show()
'''
'''
f_trace, p_trace = power(synthetic_contaminated)
f_imp, p_imp = power(imp)
f_wav, p_wav = power(wavelet)

#plt.plot(f_trace, p_trace, 'r')
#plt.show()
#plt.plot(f_imp, p_imp, 'b')
#plt.show()
#plt.plot(f_wav, p_wav, 'g')
#plt.show()
#plt.plot(synthetic, 'r')
#plt.plot(synthetic_contaminated, 'b')
#plt.show()
'''
'''
# number of spikes
no_spikes = np.random.poisson(10, 1)
# location of spikes
lo_spikes = np.random.randint(0, 200, no_spikes)
# magnitude and direction of spikes
spikes = np.random.normal(0, 0.1, no_spikes)
for a, b in zip(spikes, lo_spikes):
    rc = (b, a)
fig, ax = plt.subplots(1, 1)
x = np.arange(0, 200)
ax.plot(x, rc, 'bo', ms=8)
ax.vlines(x, 0, spikes, colors='b', lw=5, alpha=0.5)


tt = [1,4,6,7,9,3,4,6,8,6,3,7]
x = np.arange(0, len(tt))
plt.step(x, tt, label='pre (default)')
plt.plot(x, tt, 'o--', color='grey', alpha=0.3)
plt.show()
plt.plot(tt)
plt.show()
'''


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y

# wavelet parameters
f = 50
length = 0.1
dt = 1e-3  # 1ms, 1,000Hz
wavelet = ricker(f, length, dt)
rc = np.zeros(200)
rc[75] = 0.5
rc[150] = -0.25
syn = np.convolve(wavelet, rc, mode='same')
plt.plot(syn)
plt.show()
'''
wavelet = []
for f in [50, 100, 150]:
    wavelet.append(ricker(f, length, dt))
plt.subplot(1, 3, 1)
plt.plot(wavelet[0])
plt.subplot(1, 3, 2)
plt.plot(wavelet[1])
plt.subplot(1, 3, 3)
plt.plot(wavelet[2])
plt.show()

for length in [0.05, 0.1, 0.2]:
    wavelet.append(ricker(f, length, dt))
plt.subplot(1, 3, 1)
plt.plot(wavelet[3])
plt.subplot(1, 3, 2)
plt.plot(wavelet[4])
plt.subplot(1, 3, 3)
plt.plot(wavelet[5])
plt.show()

for dt in [0.5e-3, 1e-3, 1.5e-3]:
    wavelet.append(ricker(f, length, dt))
plt.subplot(1, 3, 1)
plt.plot(wavelet[6], 'r+')
plt.subplot(1, 3, 2)
plt.plot(wavelet[7], 'r+')
plt.subplot(1, 3, 3)
plt.plot(wavelet[8], 'r+')
plt.show()
'''


