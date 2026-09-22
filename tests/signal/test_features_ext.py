"""Tests of the C extension ``apollon.signal._features``."""
import os
import sys
import unittest

import numpy as np
from scipy.spatial.distance import pdist

from apollon.signal import _features, features
from apollon.signal.tools import fti16, sinusoid


def correlogram_reference(sig, delays, wlen, off_max):
    """Row ``i`` holds ``max(r, 0)**4`` for delay ``delays[i]``."""
    out = np.empty((len(delays), off_max))
    for i, delay in enumerate(delays):
        for off in range(off_max):
            r = np.corrcoef(sig[off:off+wlen],
                            sig[off+delay:off+delay+wlen])[0, 1]
            out[i, off] = max(r, 0.0)**4
    return out


def embed(sig, delay, m_dim):
    """Delay embedding of ``sig``, one vector per row."""
    n_vectors = sig.size - (m_dim-1) * delay
    return np.stack([sig[m*delay:m*delay+n_vectors] for m in range(m_dim)],
                    axis=1)


class ExtensionTestCase(unittest.TestCase):
    """Shared inputs."""

    def setUp(self):
        self.sig = np.random.default_rng(0).normal(size=50)
        # cdim_bader needs 2390 + (m_dim-1)*delay samples
        self.delay, self.m_dim = 14, 3
        self.n_min = 2390 + (self.m_dim-1) * self.delay
        snd = sinusoid((300, 600), (.2, .1), fps=3000, noise=.01)
        self.snd = fti16(snd).ravel()[:self.n_min]


class TestErrorsPropagate(ExtensionTestCase):
    """A failed conversion raises numpy's own error, not ``SystemError``."""

    def test_non_numeric_input(self):
        calls = {
            'correlogram': lambda: _features.correlogram('abc', 4, 2),
            'correlogram_delay': lambda: _features.correlogram_delay(
                'abc', [1], 4, 2),
            'emb_dists': lambda: _features.emb_dists('abc', 1, 2),
            'cdim_bader': lambda: _features.cdim_bader('abc', 1, 2, 1000, 10),
        }
        for name, call in calls.items():
            with self.subTest(name):
                with self.assertRaises((ValueError, TypeError)):
                    call()

    def test_oversized_argument(self):
        """Arguments beyond ``Py_ssize_t`` are rejected, not wrapped."""
        with self.assertRaises(OverflowError):
            _features.correlogram(self.sig, 2**70, 4)


class TestArgumentChecks(ExtensionTestCase):
    """Arguments that would make the C code read out of bounds are rejected."""

    def test_correlogram(self):
        for args in ((-1, 4), (1, 4), (8, 0), (40, 10)):
            with self.subTest(wlen=args[0], delay_max=args[1]):
                with self.assertRaises(ValueError):
                    _features.correlogram(self.sig, *args)

    def test_correlogram_largest_accepted(self):
        """``wlen + delay_max`` may reach one less than the signal length."""
        self.assertEqual(_features.correlogram(self.sig, 40, 9).shape, (9, 1))

    def test_correlogram_delay(self):
        for delays, wlen, off_max in (([1], 1, 4), ([1], 8, 0), ([], 8, 4),
                                      ([-1], 8, 4), ([39], 8, 4),
                                      ([1], 40, 11)):
            with self.subTest(delays=delays, wlen=wlen, off_max=off_max):
                with self.assertRaises(ValueError):
                    _features.correlogram_delay(self.sig, delays, wlen, off_max)

    def test_correlogram_delay_largest_accepted(self):
        """Delays may reach ``len(sig) - wlen - off_max``."""
        out = _features.correlogram_delay(self.sig, [38], 8, 4)
        self.assertEqual(out.shape, (1, 4))

    def test_emb_dists(self):
        for inp, delay, m_dim in ((self.sig, 0, 2), (self.sig, 1, 0),
                                  (np.array([]), 1, 1), (self.sig, 10, 6)):
            with self.subTest(size=inp.size, delay=delay, m_dim=m_dim):
                with self.assertRaises(ValueError):
                    _features.emb_dists(inp, delay, m_dim)

    def test_emb_dists_single_vector(self):
        """An embedding that fits exactly once has no pairs."""
        self.assertEqual(_features.emb_dists(self.sig, 7, 8).size, 0)

    def test_cdim_bader(self):
        for snd, args in ((self.snd, (0, 3, 1000, 10)),
                          (self.snd, (14, 0, 1000, 10)),
                          (self.snd, (14, 3, 1, 1)),
                          (self.snd, (14, 3, 1000, 401)),
                          (self.snd[:-1], (14, 3, 1000, 10))):
            with self.subTest(size=snd.size, args=args):
                with self.assertRaises(ValueError):
                    _features.cdim_bader(snd, *args)

    def test_cdim_short_input_raises(self):
        """``features.cdim`` surfaces the length check."""
        sig = np.zeros((self.n_min-1, 1))
        with self.assertRaises(ValueError):
            features.cdim(sig, self.delay, self.m_dim)


class TestNoLeaks(ExtensionTestCase):
    """No call keeps a reference to its input, on success or on error."""

    def assert_no_leak(self, arr, call, n_calls=100):
        before = sys.getrefcount(arr)
        for _ in range(n_calls):
            try:
                call()
            except ValueError:
                pass
        self.assertEqual(sys.getrefcount(arr), before)

    def test_correlogram(self):
        self.assert_no_leak(self.sig,
                            lambda: _features.correlogram(self.sig, 8, 4))
        self.assert_no_leak(self.sig,
                            lambda: _features.correlogram(self.sig, 40, 10))

    def test_correlogram_delay(self):
        delays = np.array([1, 3, 5], dtype=np.intp)
        for bad in (False, True):
            with self.subTest(bad=bad):
                off_max = 40 if bad else 4
                call = lambda: _features.correlogram_delay(
                    self.sig, delays, 8, off_max)
                self.assert_no_leak(self.sig, call)
                self.assert_no_leak(delays, call)

    def test_emb_dists(self):
        self.assert_no_leak(self.sig,
                            lambda: _features.emb_dists(self.sig, 2, 3))
        self.assert_no_leak(self.sig,
                            lambda: _features.emb_dists(self.sig, 10, 6))

    def test_cdim_bader(self):
        self.assert_no_leak(self.snd, lambda: _features.cdim_bader(
            self.snd, self.delay, self.m_dim, 1000, 10), n_calls=5)
        short = self.snd[:-1]
        self.assert_no_leak(short, lambda: _features.cdim_bader(
            short, self.delay, self.m_dim, 1000, 10))

    @unittest.skipUnless(sys.platform.startswith('linux'),
                         'reads the virtual size from /proc')
    def test_cdim_bader_frees_on_failed_allocation(self):
        """A histogram too large to allocate raises MemoryError, and the
        distances allocated before it, about 23 MB, are freed."""
        def vsize():
            with open('/proc/self/statm', encoding='ascii') as statm:
                return int(statm.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        before = vsize()
        for _ in range(20):
            with self.assertRaises(MemoryError):
                _features.cdim_bader(self.snd, self.delay, self.m_dim,
                                     2**60, 10)
        self.assertLess(vsize() - before, 100 * 2**20)


class TestResults(ExtensionTestCase):
    """The rebuilt argument handling passes the right data to the C code."""

    def test_correlogram_delay_matches_reference(self):
        delays = [1, 3, 5]
        out = _features.correlogram_delay(self.sig, delays, 8, 20)
        ref = correlogram_reference(self.sig, delays, 8, 20)
        self.assertTrue(np.allclose(out, ref))

    def test_emb_dists_matches_pdist(self):
        out = _features.emb_dists(self.sig, 3, 4)
        self.assertTrue(np.allclose(out, pdist(embed(self.sig, 3, 4))))

    def test_cdim_n_bins_sets_the_resolution(self):
        """Doubling n_bins halves the bin width, so doubling scaling_size
        as well keeps the range of the slope, and the estimate."""
        t = np.arange(3000) / 3000
        x = (0.2*np.sin(2*np.pi*300*t) + 0.1*np.sin(2*np.pi*600*t)
             + 0.01*np.random.default_rng(0).normal(size=3000))
        snd = fti16(x.reshape(-1, 1)).ravel()
        base = _features.cdim_bader(snd, self.delay, self.m_dim, 1000, 10)
        fine = _features.cdim_bader(snd, self.delay, self.m_dim, 2000, 20)
        self.assertAlmostEqual(fine, base, delta=0.05)

    def test_cdim_bader_is_finite(self):
        out = _features.cdim_bader(self.snd, self.delay, self.m_dim, 1000, 10)
        self.assertTrue(np.isfinite(out))


if __name__ == '__main__':
    unittest.main()
