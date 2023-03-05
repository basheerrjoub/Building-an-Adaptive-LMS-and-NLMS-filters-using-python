import numpy as np
import matplotlib.pyplot as plt

N = 2000
n = np.arange(N)
x = np.cos(0.03 * np.pi * n)

D = np.fft.fft(x)
f = np.fft.fftfreq(N, 1 / N)

plt.subplot(2, 1, 1)
plt.xlabel("Sample n")
plt.ylabel("Magnitude")
plt.title("x[n] = Cos(0.03πn)")
plt.plot(x, "g")

plt.subplot(2, 1, 2)
plt.plot(f, np.abs(D), "g")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude Spectrum")
plt.title("Magnitude Spectrum")

plt.tight_layout()
plt.show()
