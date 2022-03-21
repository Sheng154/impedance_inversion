import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import segyio

# field seismic trace
dt = 0.001
segyfile = 'ar55.2401.ew0208.hr10.migration.segy'
seisData = segyio.open(segyfile, strict=False)
field_trace = seisData.trace[1200]
section = field_trace[800:1000]
fieldSeis = section / max(section)
# plt.plot(xt, fieldSeis, 'b')
# plt.show()
print(type(fieldSeis))
plt.plot(fieldSeis)

# section2 = np.pad(section1, (0, 50), 'constant', constant_values=(0, 0))


def power(timeseries):
    fourier_transform = np.fft.rfft(timeseries)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, 1/(2*dt), len(power_spectrum))
    return frequency, power_spectrum


f_trace, p_trace = power(section)
AA = np.fft.irfft(p_trace)

#plt.plot(f_trace, p_trace, 'b')
#plt.show()




