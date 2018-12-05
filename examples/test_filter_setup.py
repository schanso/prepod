import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

import prepod.lib.prep as prep


srate = 128
duration = 1.5
l_cutoff = 10
h_cutoff = 60

# time vector
x = np.arange(0, duration, 1/srate)
# clean signal with components at 10 and 25 Hz
y_clean = np.sin(2*np.pi*10*x) + 0.5 * np.sin(2*np.pi*25*x)
# add line + Gaussian noise
y_noisy = y_clean + 0.1 * np.sin(2*np.pi*60*x) + 0.2 * np.random.normal(size=len(x))

# define baseline bandpass filter
baseline_filt = signal.firwin(400, [l_cutoff/srate, h_cutoff/srate], pass_zero=False)

# filter signal
y_filt1 = signal.convolve(y_noisy, baseline_filt, mode='same')
y_filt2 = prep.filter_raw(y_noisy, srate, b_pass=True, l_cutoff=l_cutoff, h_cutoff=h_cutoff)

# plot clean against filtered
plt.figure(figsize=(18,4))
plt.plot(x, y_noisy, alpha=.4)
plt.plot(x, y_clean, '--', c='green')
plt.plot(x, y_filt1, alpha=.5, c='orange')
plt.plot(x, y_filt2, alpha=.5, c='red')
plt.show()
