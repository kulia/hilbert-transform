import numpy as np
from numpy import sin, tan, pi

def hilbert(x):
	"""
	Compute the analytic signal of :math:`x(t)` using the discrete Hilbert transform with Hanh's method [1].
	
	The analytic signal is given by
	:math:`z(t) = x(t) + j\\frac{PV}{\pi} \\int_{-\\inft}^{\\inft} \\frac{x(t)}{t-\\tau}`
	In this function, the discrete analytic signal is estimated using
	:math:`z[n] = x[n] + j\\sum_{n=0}^{N-1} h[n-m]x[n]
	where
	:math:`h[n] = \\frac{2}{N} \\sin^2(\frac{\pi n}{2}) \\cot(\frac{\\pi n}{N})`

	[1] S. Hanh, Hilbert Transform in Signal Processing, Artech House, Boston 1996

	:param x: array_like
	:return z: ndarray
			Analytic signal of x
	"""

	N = len(x)

	np.seterr(divide='ignore')

	cot = lambda theta: np.nan_to_num(1 / tan(theta))

	n = np.arange(2*N)
	h =  (2/N) * cot(n*pi/N) * (sin(n*pi/2)**2)

	y = np.convolve(x, h, mode='valid')

	return x + 1j*y[-len(x):]