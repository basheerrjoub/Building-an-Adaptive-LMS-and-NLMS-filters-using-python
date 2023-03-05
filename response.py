import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

b_d = [1, -2, 4]

# Define the frequency range
w = np.linspace(-np.pi, np.pi, 2000)

H_d = np.zeros(w.shape, dtype=complex)
for i in range(len(b_d)):
    H_d += b_d[i] * np.exp(-1j * i * w)
w = w / (np.pi)
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(w, np.abs(H_d), "g")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Amplitude")
plt.title("Amplitude Response")
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(w, np.angle(H_d), label="d")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (rad)")
plt.title("Phase Response")

plt.grid()
plt.tight_layout()
plt.show()
