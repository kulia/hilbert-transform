import unittest
import numpy as np
from numpy.linalg import norm
from hilbert import hilbert

from scipy.signal import hilbert as scipy_hilbert

class TestHilbert(unittest.TestCase):
	def test_cosine(self):
		N = 1000
		n = np.arange(1000)
		t = n/N
		theta = 2*np.pi*t
		x = np.cos(theta)

		y_hat = np.real(hilbert(x))
		y = np.sin(theta)

		def snr(y, y_hat):
			return norm(y) / norm(y_hat - y)

		''' Allow small error '''
		self.assertTrue(snr(y, y_hat) < 1)

	def test_scipy(self):
		x = np.random.rand(1000)
		h_1 = hilbert(x)

		h_2 = scipy_hilbert(x)

		''' Allow small error '''
		self.assertTrue(norm(h_1-h_2), 1e-6)

if __name__ == '__main__':
	unittest.main()