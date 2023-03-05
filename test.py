import numpy as np
import matplotlib.pyplot as plt

# Define the coefficients of the system
b = [1, -2, 4]

# Define the frequency range
w = np.linspace(-np.pi, np.pi, 2000)

# Calculate the frequency response
H = np.zeros(w.shape, dtype=complex)
for i in range(len(b)):
    H += b[i] * np.exp(-1j * i * w)

# Calculate the amplitude response
A = np.abs(H)

# Calculate the phase response
P = np.angle(H)
w = w / (np.pi)
# Plot the amplitude response
plt.plot(w, A)

plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Amplitude")
plt.title("Amplitude Response")
plt.grid()
plt.show()

# Plot the phase response
plt.plot(w, P)
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (rad)")
plt.title("Phase Response")
plt.grid()
plt.show()
