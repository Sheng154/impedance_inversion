import matplotlib.pyplot as plt
import numpy as np

f = 500
length = 0.001  # 1 ms
dt = 1e-3

tb = [250, 200, 150, 100, 50, 25]
tb_loc = np.array([4, 5, 6, 7, 8, 9])
tb_thick = 2e-4  # ms
interval = 1e-3  # ms
sample_f = 2e4  # Hz  0.01 ms  \ 100 samples per ms
tb_len = int(tb_thick * sample_f)
tb_sample = np.asarray(tb_loc * 1e-3 * sample_f, dtype=int)

L0 = int(3e-3 * sample_f)
L1 = int(7e-3 * sample_f)
imp_L0 = 1500 * np.ones((L0, 1))
imp_L1 = 2000 * np.ones((L1, 1))
imp_trend = np.concatenate((imp_L0, imp_L1))
imp = np.concatenate((imp_L0, imp_L1))
for i in range(len(tb_loc)):
    imp[tb_sample[i]:tb_sample[i]+tb_len] = imp_trend[tb_sample[i]:tb_sample[i]+tb_len] - tb[i]
imps = np.reshape(imp, (len(imp), ))


def calcuRc(imp):
    Rc = []
    nsamp = np.shape(imp)[0]
    for i in range(0, nsamp - 1):
        Rc.append((imp[i + 1] - imp[i]) / (imp[i + 1] + imp[i]))
    rc = np.asarray(Rc)
    return rc


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t0 ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t0 ** 2))
    return y


def SyntheticTrace(Rc, wavlt):
    synthetic_trace = np.convolve(Rc, wavlt, mode='same')
    synthetic_norm = synthetic_trace / max(synthetic_trace)
    return synthetic_trace, synthetic_norm


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1 / (2 * dt), len(power_spectrum))
    return frequency, power_spectrum


rc = calcuRc(imps)
wavelet = ricker(f, length, dt)
Synthetic_raw, model_trace = SyntheticTrace(rc, wavelet)


plt.plot(imp)
plt.ylabel('Acoustic impedance')
plt.xlabel('Time series')
plt.show()

plt.plot(rc)
plt.ylabel('Reflectivity coefficient')
plt.xlabel('Time series')

plt.plot(wavelet)
plt.ylabel('Amplitude')
plt.xlabel('Time series')

plt.plot(model_trace)
plt.ylabel('Amplitude')
plt.xlabel('Time series')
plt.show()


f_trace, p_trace = power(model_trace)
f_Rc, p_Rc = power(rc)
f_wav, p_wav = power(wavelet)

# imp = np.reshape(imp, (1, 1000))
'''
plt.subplot(1, 3, 1)
plt.plot(f_wav, p_wav, 'g')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Wavelet')
# plt.show()
plt.subplot(1, 3, 2)
plt.plot(f_Rc, p_Rc, 'b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Reflectivity series')
# plt.show()
plt.subplot(1, 3, 3)
plt.plot(f_trace, p_trace, 'purple')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Seismic trace')
plt.show()
'''
