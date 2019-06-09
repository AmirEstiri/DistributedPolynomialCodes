import numpy as np


def recursive_fft(y):
    n = len(y)
    a = np.zeros(n, dtype=complex)
    if n == 1:
        return y
    w = 1
    y0 = y[::2]
    y1 = y[1::2]
    a0 = recursive_fft(y0)
    a1 = recursive_fft(y1)
    for k in range(int(n / 2)):
        a[k] = a0[k] + w * a1[k]
        a[k + int(n / 2)] = a0[k] - w * a1[k]
        w = w * np.exp(-2 * np.pi * 1j / n)
    return a


value = np.array([2.0, -2.8+3.8*1j, -6.4, -2.8-3.8*1j])
# value = np.array([-2.8+3.8*1j, -6.4, -2.8+4.2*1j, 2.0])
print(recursive_fft(value) / 4)
