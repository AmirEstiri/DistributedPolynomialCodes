import numpy as np


def recursive_fft(y):
    n = len(y)
    a = np.zeros(n)
    if n == 1:
        return y
    w = 1
    y0 = y[::2]
    y1 = y[1::2]
    a0 = recursive_fft(y0)
    a1 = recursive_fft(y1)
    for k in range(int(n / 2 - 1)):
        a[k] = a0[k] + w * a1[k]
        a[k + int(n / 2)] = a0[k] - w * a1[k]
        w = w * np.exp(-2 * np.pi * 1j / n)
    return y


coeffs = np.array([0.1, 0.3, 0.6, 0.0, 0.9, 5])
x = np.array([1, 2, 3, 4, 5, 6])
value = np.array([6.9, 19.6, 72.5, 226.2, 584.5, 1306.4])

# z = np.zeros(5)
z = np.fft.ifft(value) / len(value)
print(z)
print(recursive_fft(value))
