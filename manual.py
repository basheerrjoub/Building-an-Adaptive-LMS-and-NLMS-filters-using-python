import numpy as np
import matplotlib.pylab as plt


def lms(x, dn, mu, M, N):
    N = N
    w = np.zeros(M)
    w1 = np.zeros((N, M))
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x1 = np.flip(x[n - M + 1 : n + 1])
        # y[n] = np.dot(w, x1)
        y[n] = sum(w[i] * x1[i] for i in range(M))
        e[n] = dn[n] - y[n]
        # w = w + 2 * mu * e[n] * x1
        for i in range(M):
            w[i] = w[i] + 2 * mu * e[n] * x1[i]
        w1[n] = w

    J = e**2
    return w, y, e, J, w1


def nlms(x, dn, mu, M, N):
    N = N
    w = np.zeros(M)
    w1 = np.zeros((N, M))
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x1 = np.flip(x[n - M + 1 : n + 1])
        y[n] = sum(w[i] * x1[i] for i in range(M))
        e[n] = dn[n] - y[n]
        x1_norm = np.linalg.norm(x1)
        if x1_norm != 0:
            w = w + (2 * mu * e[n] / x1_norm) * x1
        w1[n] = w
    J = e**2
    return w, y, e, J, w1


def awgn(x, snr, signalpower):
    # Calculate the noise power
    noise_power = signalpower / snr

    # Generate Gaussian noise
    noise = np.random.normal(scale=np.sqrt(noise_power), size=x.shape)

    # Add the noise to the signal
    return x + noise


N = 2000
n = np.arange(N)
x_n = np.cos(0.03 * np.pi * n)
# Add white Gaussian noise to the signal with a SNR of 20 dB
snr = 40
signalpower = np.mean(np.power(x_n, 2))
# x_n = awgn(x_n, snr, signalpower)
# Plot Noise
# plt.plot(n, x_n)
# plt.show()
# plt.close()
# exit()

d_n = x_n - (2 * np.roll(x_n, 1)) + (4 * np.roll(x_n, 2))
print(d_n[:10])
# LMS
# Initialize
M = 4  # Filter length
mu = 0.01  # learning rate
# Single iteration
w, y, e, J, w1 = nlms(x_n, d_n, mu, M, N)

# 1000 Trails
# all = []
# y_list = []
# for i in range(1000):
#     w, y, e, J, w1 = lms(x_n, d_n, mu, M)
#     J_dB = 10 * np.log10(J)
#     all.append(J_dB)
#     y_list.append(y)
# average = np.mean(all, axis=0)
# y_list = np.mean(y_list, axis=0)
# # Plot Average error
# plt.title("Error")
# plt.xlabel("Iteration n")
# plt.plot(average, "r")
# plt.show()
# plt.close()
# # Plot the adaption graph
# plt.title("Adaptation")
# plt.xlabel("Iteration n")

# # plt.plot(d_n, "r", label="d: Target Signal")
# plt.plot(y_list, "g", label="y: Adaptive Filter")
# plt.show()
# plt.close()
# exit()

for i in range(len(w1[-1])):
    print(f"W{i}: {w1[-1][i]}")

colors = ["m", "r", "b", "c", "y", "g", "k"]

# Plot the adaption graph
plt.title("Adaptation")
plt.xlabel("Iteration n")
plt.plot(d_n, "r", label="d: Target Signal")
plt.plot(y, "g", label="y: Adaptive Filter")
plt.legend()
plt.show()
plt.close()

# Plot Error Graph
plt.subplot(3, 1, 1)
plt.title("Error")
plt.xlabel("Iteration n")
plt.plot(e, "m")


# Plot Error Squared Graph
plt.subplot(3, 1, 2)
plt.title("Error[Squared]")
plt.xlabel("Iteration n")
plt.plot(J, "k")


# Calculate J in dB scale
J_dB = 10 * np.log10(J)

# Plot J_dB Graph
plt.subplot(3, 1, 3)
plt.title("Error[Squared] in dB")
plt.xlabel("Iteration n")
plt.plot(J_dB, "k")
plt.tight_layout()
plt.show()
plt.close()

# Plot filter Ciefficients
plt.title("Filter Coefficients")
plt.xlabel("Iteration n")
plt.ylabel("Value")
for i in range(M):
    plt.plot([wi[i] for wi in w1], colors[i], label=f"w{i}")
plt.legend()
plt.show()

plt.close()

# Define the coefficients of the system
b_d = [1, -2, 4]
b_y = w1[-1]
# Define the frequency range
w = np.linspace(-np.pi, np.pi, 2000)

# Calculate the frequency response
H_y = np.zeros(w.shape, dtype=complex)
H_d = np.zeros(w.shape, dtype=complex)
for i in range(len(b_d)):
    H_d += b_d[i] * np.exp(-1j * i * w)
for i in range(len(b_y)):
    H_y += b_y[i] * np.exp(-1j * i * w)
w = w / (np.pi)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, np.abs(H_d), label="d")
plt.plot(w, np.abs(H_y), label="y")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Amplitude")
plt.title("Amplitude Response")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w, np.angle(H_d), label="d")
plt.plot(w, np.angle(H_y), label="y")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (rad)")
plt.title("Phase Response")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.close()

exit()
