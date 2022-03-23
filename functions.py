import numpy as np
from scipy import signal
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from SYnthetic_case import mtrace, mtrace_norm, low_filtered_imp, dt
import pylops


def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


def createModel(nsamp):
    imp = np.random.normal(2300, 350, nsamp)  # Determines the range of synthetic impedance model
    # imp = np.random.uniform(1500, 2800, nsamp)
    imp_smoothed = ndi.uniform_filter1d(imp, size=3)
    return imp, imp_smoothed


def calibrate(imp_pop, imp_seabed):
    for ind in imp_pop:
        diff = ind[0] - imp_seabed
        for i in range(len(ind)):
            ind[i] -= diff
    return imp_pop


def calibrating(imp_pop):
    for ind in imp_pop:
        for k in range(len(ind)):
            ind[k] = ind[k] + low_filtered_imp[k]
    return imp_pop


def stepped(imp_pop):
    l = len(imp_pop[0])
    N = l // 5
    for ind in imp_pop:
        for i in range(N):
            for j in range(5):
                ind[i * 5 + j] = ind[i * 5]
    return imp_pop


def calcuRc(imp_pop):
    rc_pop = []
    if len(np.shape(imp_pop)) == 1:
        nsamp = len(imp_pop)
        rc = []
        imp_individual = imp_pop
        for j in range(nsamp - 1):
            rc.append((imp_individual[j+1] - imp_individual[j]) / (imp_individual[j+1] + imp_individual[j]))
        rc_pop.append(rc)
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
    synthetic_pop = []
    for i in range(np.shape(rc_pop)[0]):
        trace = np.convolve(rc_pop[i], wvlt, mode='same')
        trace_norm = trace / max(abs(trace))
        synthetic_pop.append(trace_norm)
    return synthetic_pop  # normalised synthetic trace population


def SyntheticTrace(Rc, wavelet, nsamp):
    noise = np.random.normal(0, 0.3, nsamp - 1)
    noise_smoothed = ndi.uniform_filter1d(noise, size=7)  # Controls the noise level
    synthetic_trace = np.convolve(Rc, wavelet, mode='same')
    synthetic_norm = synthetic_trace / max(synthetic_trace)
    synthetic_ctm = synthetic_norm + noise_smoothed
    return synthetic_trace, synthetic_norm, synthetic_ctm


def SyntheticConvmtx(wavelet, imp_pop):
    syn_pop = []
    omtx = pylops.avo.poststack.PoststackLinearModelling(wavelet / 2, nt0=len(imp_pop[0]), explicit=True)
    for ind in imp_pop:
        ind = np.asarray(ind)
        seismic = omtx * ind
        seismic = seismic / max(abs(seismic))
        syn_pop.append(seismic)
    return syn_pop


def rankFitness(synthetic_norm, trace_norm):
    fitness = {}
    error = []
    popSize = len(synthetic_norm)
    synthetic_norm = np.asarray(synthetic_norm)
    trace_norm = np.asarray(trace_norm)
    for i in range(synthetic_norm.shape[0]):
        diff = synthetic_norm[i] - trace_norm
        error.append(sum(abs(diff)))  # 该条trace的总residual
    for j in range(len(error)):
        fitness[j] = sum(error) / error[j]
    fit = fitness.values()
    ave_fitness = sum(fit) / popSize
    return ave_fitness, error, sorted(fitness.items(), key=lambda item: item[1], reverse=True)


def evaluate(synthetic, rc):
    # synthetic type: ndarray
    # rc type: list
    # imp type: deap.creator.individual
    # imp_diff = abs(np.asarray(imp) - np.asarray(low_filtered_imp))
    # trend_error = sum(imp_diff)

    rc_abs = [abs(ele) for ele in rc]
    spikes_sum = sum(rc_abs)

    diff = synthetic - mtrace_norm
    error = sum(abs(diff))
    fitness = (1/error, 1/spikes_sum)
    return fitness


def erroreval(syn_imp, imp):
    diff = []
    for i in range(len(syn_imp)):
        diff.append(abs((syn_imp[i] - imp[i]) / imp[i]))
    residual = sum(diff)/len(diff)   # average percentage error of impedance model as used by Vardy(2015)
    return residual


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1 / (2 * dt), len(power_spectrum))
    return frequency, power_spectrum


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass", analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=10):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

