from typing import List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
'''
from main_1D_impedance_inversion_GA import *
from main_1D_impedance_inversion_GA import bestmodel_smoothed, wavelet
from Synthetic_case_2 import imp


def calrc(imp):
    Rc = []
    for i in range(0, len(imp)-1):
        Rc.append((imp[i + 1] - imp[i]) / (imp[i + 1] + imp[i]))
    return Rc


Rc_imp = calrc(imp)
Rc_best = calrc(bestmodel_smoothed)
s1 = np.convolve(Rc_imp, wavelet, mode='same')
s2 = np.convolve(Rc_best, wavelet, mode='same')
s1_norm = s1 / max(s1)
s2_norm = s2 / max(s2)
#plt.plot(imp)
plt.plot(bestmodel_smoothed)
plt.ylabel('Acoustic Impedance')
plt.xlabel('Time')
#plt.show()

plt.plot(s1_norm)
plt.plot(s2_norm)
#plt.show()
'''
