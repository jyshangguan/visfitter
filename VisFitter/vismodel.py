import numpy as np
from scipy.integrate import quad
import scipy.special as special

__all__ = ["Gaussian", "vis_gauss", "vis_point"]

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


def vis_gauss(u, v, sigma_lm, A=1., i=0., pa=0., l0=0., m0=0.):
    """
    The visibility model of a Gaussian profile, which is inclined to the line of
    sight with the inclination angle i and position angle pa.  The coordinates
    transformation is consistent with 2017ApJ...839...99P.

    Parameters
    ----------
    u : array_like
        The u coordinates.
    v : array_like
        The v coordinates.
    sigma_lm : float
        The sigma of a Gaussian profile of the source image.
    A : float, default: 1.
        The amplitude of the visibility.
    i : float, default: 0.
        The inclination angle, units: radian.
    pa : float, default: 0.
        The position angle (to the east of the north), units: radian.
    l0 : float, default: 0.
        The l coordinate of the center of the model.
    m0 : float, default: 0.
        The m coordinate of the center of the model.

    Returns
    -------
    F_uv : float
        The visibility at given uv coordinates.
    """
    up = u * np.cos(pa) + v * np.sin(pa)
    vp = v * np.cos(pa) - u * np.sin(pa)
    rho = np.sqrt((up * np.cos(i))**2. + vp**2.)
    sigma_uv = 0.5 / pi / sigma_lm
    F_uv = Gaussian(rho, A, sigma_uv) * np.exp(-2. * pi * 1j * (u * l0 + v * m0))
    return F_uv


def vis_point(u, v, A=1., l0=0., m0=0.):
    """
    The visibility model of a point source.

    Parameters
    ----------
    u : array_like
        The u coordinates.
    v : array_like
        The v coordinates.
    A : float, default: 1.
        The amplitude of the visibility.
    l0 : float, default: 0.
        The l coordinate of the center of the model.
    m0 : float, default: 0.
        The m coordinate of the center of the model.

    Returns
    -------
    F_uv : float
        The visibility at given uv coordinates.
    """
    F_uv = A * np.exp(-2. * pi * 1j * (u * l0 + v * m0))
    return F_uv


def vis_symmetric_model(u, v, i=0., pa=0., l0=0., m0=0., model="Gaussian", kw_model={}):
    """
    Calculate the visibility of a symmetric model.  The coordinates transformation
    is consistent with 2017ApJ...839...99P.

    Parameters
    ----------
    u : float
        The u coordinates.
    v : float
        The v coordinates.
    i : float, default: 0
        The inclination angle, units: radian.
    pa : float, default: 0
        The position angle (to the east of the north), units: radian.
    l0 : float, default: 0
        The l coordinate of the center of the model.
    m0 : float, default: 0
        The m coordinate of the center of the model.
    model : string
        The model of the source emission radial profile.
    kw_model : dict
        The model keywords.

    Returns
    -------
    F_rho : float
        The visibility at given uv.
    """
    up = u * np.cos(pa) + v * np.sin(pa)
    vp = v * np.cos(pa) - u * np.sin(pa)
    rho = np.sqrt((up * np.cos(i))**2. + vp**2.)
    if model == "Gaussian":
        Ifunc  = lambda r: Gaussian(r, **kw_model)
    else:
        raise ValueError("The model ({0}) is not recognized!".format(model))
    j0func = lambda r: special.jv(0, 2*np.pi*rho * r)
    integral = lambda r: Ifunc(r) * j0func(r) * r
    F_rho = 2 * np.pi * quad(integral, 0, np.inf)[0] * np.exp(-2. * pi * 1j * (u * l0 + v * m0))
    return F_rho
