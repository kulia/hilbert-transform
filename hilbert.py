import numpy as np
import matplotlib.pylab as plt

from scipy.signal import hilbert as scipy_hilbert
from numpy import sin, cos, tan, pi


def hilbert(x):
	"""
	Compute the analytic signal, using the Hilbert transform with Hanh's method [1].

	[1] S. Hanh, Hilbert Transform in Signal Processing, Artech House, Boston 1996

	:param x: array_like
	:return xa: ndarray
			Analytic signal of x
	"""

	N = len(x)

	if N%2 != 0: N += 1	# Ensure even N

	cot = lambda theta: 1 / tan(theta)

	n = np.arange(2*N)
	h =  (2/N) * cot(n*pi/N) * (sin(n*pi/2)**2)

	y = np.convolve(x[:N], h, mode='valid')

	if len(y) != len(x):
		print('Hello!')
		y = y[1:]

	return x + 1j*y