import numpy as np

import scipy.signal
import scipy.stats

from spectrum import arburg

__all__ = [
    'mad', 'sma', 'energy', 'autoreg', 'corr', 'td_entropy', 'fd_entropy', 'mean_freq', 'bands_energy',
    't_feat', 'f_feat'
]

"""
TIME DOMAIN FEATURES
"""


def mad(data, axis):
    return np.median(np.abs(data), axis=axis)


def sma(data, axis):
    return np.abs(data).sum(
        tuple(np.arange(1, data.ndim))
    )[:, None]


def energy(data, axis):
    return np.power(data, 2).mean(axis=axis)


def autoreg(data, axis):
    def _autoreg(datum):
        order = 4
        try:
            coef, _, _ = arburg(datum, order)
            coef = coef.real.tolist()
        except ValueError:
            coef = [0] * order
        return coef
    
    ar = np.asarray([
        [_autoreg(data[jj, :, ii]) for ii in range(data.shape[2])] for jj in range(data.shape[0])
    ])
    
    return ar.reshape(ar.shape[0], -1)


def corr(data, axis):
    inds = np.tril_indices(3, k=-1)
    cor = np.asarray([
        np.corrcoef(datum.T)[inds] for datum in data
    ])
    return cor


def td_entropy(data, axis, bins=16):
    bins = np.linspace(-4, 4, bins)
    
    def _td_entropy(datum):
        ent = []
        for ci in range(datum.shape[1]):
            try:
                pp, bb = np.histogram(datum[:, ci], bins, density=True)
                ent.append(scipy.stats.entropy(pp * (bb[1:] - bb[:-1]), base=2))
            except ValueError as ex:
                raise ex
        return ent
    
    H = np.asarray([
        _td_entropy(datum) for datum in data
    ])
    
    return H


"""
FREQUENCY DOMAIN FEATURES
"""


def fd_entropy(psd, axis, td=False):
    H = scipy.stats.entropy(
        (
            psd / psd.sum(axis=axis)[:, None, :]
        ).transpose(1, 0, 2), base=2
    )
    return H


def mean_freq(freq, spec, axis):
    return (spec * freq[None, :, None]).sum(axis=axis)


def bands_energy(freq, spec, axis):
    # Based on window of 2.56 seconds sampled at 50 Hz: 128 samples
    orig_freqs = np.fft.fftfreq(128, 1 / 50)[:64]
    orig_band_inds = np.asarray([orig_freqs[[ll - 1, uu - 1]] for ll, uu in [
        [1, 8], [9, 16], [17, 24], [25, 32],
        [33, 40], [41, 48], [49, 56], [57, 64],
        [1, 16], [17, 32], [22, 48], [49, 64],
        [1, 24], [25, 48]
    ]])
    
    # Generate the inds
    bands = np.asarray([(freq > ll) & (freq <= uu) for ll, uu in orig_band_inds]).T
    
    # Compute the sum with tensor multiplication
    band_energy = np.einsum(
        'ijk,kl->ijl', spec.transpose(0, 2, 1), bands
    ).transpose(0, 2, 1)
    band_energy = band_energy.reshape(band_energy.shape[0], -1)
    
    return band_energy


def add_magnitude(data):
    assert isinstance(data, np.ndarray)
    return np.concatenate((data, np.sqrt(np.sum(data ** 2, axis=2, keepdims=True) - 1)), axis=2)


"""
Time and frequency feature interfaces
"""


def t_feat(key, index, data, fs):
    data = add_magnitude(data)
    features = [
        f(data, axis=1) for f in [
            np.mean,  # 3 (cumsum: 3)
            np.std,  # 3 (cumsum: 6)
            mad,  # 3 (cumsum: 9)
            np.max,  # 3 (cumsum: 12)
            np.min,  # 3 (cumsum: 15)
            sma,  # 1 --- (cumsum: 16)
            energy,  # 3 --- (cumsum: 19)
            scipy.stats.iqr,  # 3 (cumsum: 22)
            td_entropy,  # 3 (cumsum: 25)
            # autoreg,  # 12 (cumsum: 37)
            corr,  # 3 (cumsum: 40)
        ]
    ]
    
    return np.concatenate(features, axis=1)


def f_feat(key, index, data, fs):
    data = add_magnitude(data)
    
    freq, spec = scipy.signal.periodogram(data, fs=fs, axis=1)
    spec_normed = spec / spec.sum(axis=1)[:, None, :]
    
    features = [
        f(spec, axis=1) for f in [
            np.mean,  # 3 (cumsum: 3)
            np.std,  # 3 (cumsum: 6)
            mad,  # 3 (cumsum: 9)
            np.max,  # 3 (cumsum: 12)
            np.min,  # 3 (cumsum: 15)
            sma,  # 1 (cumsum: 16)
            energy,  # 3 (cumsum: 19)
            scipy.stats.iqr,  # 3 (cumsum: 22)
            fd_entropy,  # 3 (cumsum: 25)
            np.argmax,  # 3 (cumsum: 28)
            scipy.stats.skew,  # 3 (cumsum: 31)
            scipy.stats.kurtosis,  # 3 (cumsum: 34)
        ]
    ]
    
    features += [
        mean_freq(freq, spec_normed, axis=1),  # 3 (cumsum: 37)
        bands_energy(freq, spec_normed, axis=1),  # 42 (cumsum: 79) (not on mag)
    ]
    
    return np.concatenate(features, axis=1)
