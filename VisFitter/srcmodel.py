import numpy as np

__all__ = ["src_gauss"]

pi = np.pi


def Gaussian(x, A, sigma):
    """
    The Gaussian function

    Parameters
    ----------
    x : array like
        The independent variable.
    A : float
        The amplitude.
    sigma : float
        The sigma of a Gaussian profile.

    Returns
    -------
    y : array like
        The dependent variable.
    """
    x = np.atleast_1d(x)
    y = A * np.exp(-0.5 * x**2 / sigma**2)
    return y


def src_gauss(l, m, sigma_lm, A=1., i=0., pa=0., l0=0., m0=0.):
    """
    A 2D Gaussian model on the source emission.

    Parameters
    ----------
    l : array like
        The l axis coordinate.
    m : array like
        The m axis coordinate.
    sigma_lm : float
        The sigma of a Gaussian profile of the source image.
    A : float, default: 1.
        The amplitude of the visibility.
    i : float, default: 0.
        The inclination angle, units: radian.
    pa : float, default: 0.
        The position angle, units: radian.
    l0 : float, default: 0.
        The l coordinate of the center of the model.
    m0 : float, default: 0.
        The m coordinate of the center of the model.

    Returns
    -------
    I : array like
        The intensity of the source emission.
    """
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    sigma_x = sigma_lm
    sigma_y = sigma_lm * np.cos(i)
    a = 0.5 * ((np.cos(pa) / sigma_x)**2. + (np.sin(pa) / sigma_y)**2.)
    b = 0.5 * np.sin(2. * pa) * (sigma_x**-2. - sigma_y**-2.)
    c = 0.5 * ((np.sin(pa) / sigma_x)**2. + (np.cos(pa) / sigma_y)**2.)
    p = a * (l - l0)**2. + b * (l - l0) * (m - m0) + c * (m - m0)**2.
    I = A * np.exp(-p) / (2. * pi * sigma_x * sigma_y)
    return I
