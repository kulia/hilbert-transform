import numpy as np
from numpy import sin, tan, pi

def hilbert(x):
	"""
	Compute the analytic signal, using the Hilbert transform with Hanh's method [1].

	[1] S. Hanh, Hilbert Transform in Signal Processing, Artech House, Boston 1996

	:param x: array_like
	:return xa: ndarray
			Analytic signal of x
	"""

	N = len(x)

	cot = lambda theta: 1 / tan(theta)

	n = np.arange(2*N)
	h =  (2/N) * cot(n*pi/N) * (sin(n*pi/2)**2)

	y = np.convolve(x, h, mode='valid')

	return x + 1j*y[-len(x):]