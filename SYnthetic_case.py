import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import scipy.ndimage as ndi
from scipy import signal
import pandas as pd
import random
from scipy.signal import filtfilt
import pylops
# import segyio from functions import *

tmax = 200 * 1e-3  # 10ms
tmin = 0
xt = np.arange(0, 200)

# impedance range
max_guess, min_guess = 4500, 1000
max_imp, min_imp = 3000, 1500
# wavelet parameters
f = 1000
length = 0.071
dt = 1e-3  # 1ms, 1,000Hz
# Optimisation criterion
L1 = True
###########################
np.random.seed(12)

nsamp = int((tmax - tmin) / dt)


def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


def createModel(nsamp):
    nsmooth = 3
    imp = np.random.normal(2300, 400, nsamp)  # Determines the range of synthetic impedance model
    imp_filt = filtfilt(np.ones(nsmooth) / float(nsmooth), 1, imp)
    imp_trend = imp_filt + np.arange(nsamp)
    imp_smooth = ndi.uniform_filter1d(imp, size=3)
    return imp_trend


def calcuRc(imp_pop):
    rc_pop = []
    if len(np.shape(imp_pop)) == 1:
        nsamp = len(imp_pop)
        imp_individual = imp_pop
        for j in range(nsamp - 1):
            rc_pop.append((imp_individual[j+1] - imp_individual[j]) / (imp_individual[j+1] + imp_individual[j]))
    else:
        nmodel = np.shape(imp_pop)[0]
        nsamp = np.shape(imp_pop)[1]
        for i in range(nmodel):
            rc = []
            imp_individual = imp_pop[i]
            for j in range(nsamp - 1):
                rc.append((imp_individual[j+1] - imp_individual[j]) / (imp_individual[j+1] + imp_individual[j]))
            rc_pop.append(rc)
    return rc_pop


def generateSynthetic(rc_pop, wvlt):
    trace = np.convolve(rc_pop, wvlt, mode='same')
    trace2 = np.convolve(rc_pop, wvlt, mode='full')
    trace3 = np.convolve(rc_pop, wvlt, mode='validate')
    trace_norm = trace / max(abs(trace))
    return trace_norm # trace_norm  # normalised synthetic trace population


def erroreval(syn_imp, imp):
    diff = []
    for i in range(len(syn_imp)):
        diff.append(abs((syn_imp[i] - imp[i])) / imp[i])
    residual = sum(diff)/len(diff)   # average percentage error of impedance model as used by Vardy(2015)
    return residual


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1 / (2 * dt), len(power_spectrum))
    plt.plot(frequency, power_spectrum)
    plt.show()


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    return b, a


def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="highpass")
    return b, a


def butter_highpass_filter(data, cutoff, fs):
    b, a = butter_highpass(cutoff, fs, order=5)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs, order=5)
    y = signal.filtfilt(b, a, data)
    return y


wavelet = ricker(f, length, dt)
imp = createModel(nsamp)
imp_log = np.log(imp)
mback = filtfilt(np.ones(int(len(imp_log)/20)) / float(int(len(imp_log)/20)), 1, imp_log)
omtx = pylops.avo.poststack.PoststackLinearModelling(wavelet/2, nt0=len(imp), explicit=True)
mtrace = omtx * imp
mtrace = mtrace/max(mtrace)
mtrace_n = mtrace + np.random.normal(0, 1e-2, mtrace.shape)
Rc = calcuRc(imp)
model_trace = generateSynthetic(Rc, wavelet)

high_filtered_imp = butter_highpass_filter(imp, 50, 1000)
low_filtered_imp = butter_lowpass_filter(imp, 50, 1000)

minv1 = pylops.avo.poststack.PoststackInversion(
    mtrace_n, wavelet/2, m0=mback, explicit=True, simultaneous=True)[0]
minv = pylops.avo.poststack.PoststackInversion(
    mtrace, wavelet/2, m0=mback, explicit=True, simultaneous=True)[0]
'''
plt.subplot(3, 1, 1)
plt.plot(imp, label='raw')
plt.plot(model_trace, label='smooth')
plt.legend(loc="upper left")
plt.subplot(3, 1, 2)
plt.plot(Rc, label='raw')
plt.legend(loc="upper left")
plt.subplot(3, 1, 3)
f_Rc, p_Rc = power(Rc)
plt.plot(f_Rc, p_Rc, label='raw')
plt.legend(loc="upper left")
plt.show()

f_imp, p_imp = power(imps)
f_trace, p_trace = power(model_trace)
f_Rc, p_Rc = power(Rc)
f_wav, p_wav = power(wavelet)
plt.subplot(3, 1, 1)
plt.plot(model_trace)
plt.subplot(3, 1, 2)
plt.plot(f_Rc, p_Rc, 'b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Reflectivity series')
plt.subplot(3, 1, 3)
plt.plot(f_imp, p_imp)
plt.show()

plt.subplot(3, 1, 1)
plt.plot(f_wav, p_wav, 'g')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Wavelet')
# plt.show()
plt.subplot(3, 1, 2)
plt.plot(f_Rc, p_Rc, 'b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Reflectivity series')
# plt.show()
plt.subplot(3, 1, 3)
plt.plot(f_trace, p_trace, 'purple')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Seismic trace')
# plt.show()
plt.plot(f_trace, p_trace, 'purple')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Seismic trace')
plt.show()
'''

'''
plot_x = np.arange(1, 200 + 1)
fig, ax = plt.subplots()
ax.plot(plot_x, imps, label='Synthetic impedance model', linewidth=4)
ax.plot(plot_x, low_filtered_imp, label='Low frequency trend', linewidth=4)
ax.tick_params(direction='out', length=15, width=4, grid_color='r', grid_alpha=0.5)
# ax.spines[bottom].set_linewidth(size).
mpl.rcParams['axes.linewidth'] = 2 #set the value globally
ax.set_xlim(0, 200)
ax.set_ylim(1500, 2800)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
ax.spines['bottom'].set_linewidth(4)
ax.spines['top'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Acoustic impedance $\mathregular{(m/s · g/cm^{3})}$', fontsize=32)
plt.legend(fontsize=32)
# plt.plot(imps, 'r')
# plt.plot(imps_, 'b')
plt.show()
'''


'''
plt.plot(imps, 'g')
#plt.plot(high_filtered_imp)
plt.plot(low_filtered_imp)
plt.xlabel('time')
plt.ylabel('Impedance')
plt.show()
f_imp_, p_imp_ = power(high_filtered_imp)
f_imp_l, p_imp_l = power(low_filtered_imp)
plt.plot(f_imp_, p_imp_, 'g')
plt.plot(f_imp_l, p_imp_l, 'r')
plt.show()
'''
