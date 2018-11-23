import acor
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import truncnorm
from time import time
PI2 = 2. * np.pi

__all__ = ["EmceeModel"]

#The probability functions

def lnprior(params, data, model):
    """
    Calculate the ln prior probability.
    """
    lnprior = 0.0
    parIndex = 0
    parDict = model.get_modelParDict()
    for modelName in model._modelList:
        parFitDict = parDict[modelName]
        for parName in parFitDict.keys():
            if parFitDict[parName]["vary"]:
                parValue = params[parIndex]
                parIndex += 1
                pr1, pr2 = parFitDict[parName]["range"]
                if (parValue < pr1) or (parValue > pr2):
                    lnprior -= np.inf
            else:
                pass
    return lnprior

def ChSq(data, model, unct=None):
    '''
    This function calculate the Chi square of the observed data and
    the model.

    Parameters
    ----------
    data : float array
        The observed data.
    model : float array
        The model.
    unct : float array
        The uncertainties.

    Returns
    -------
    chsq : float
        The Chi square

    Notes
    -----
    None.
    '''
    if unct is None:
        unct = np.ones_like(data)
    wrsd = (data - model)/unct #The weighted residual
    chsq = np.sum(wrsd**2) + np.sum( np.log(PI2 * unct**2) )
    return chsq

def lnlike_amp(params, data, model):
    """
    Calculate the ln likelihood using only the amplitude data.
    """
    visa = data["visamp"]
    visae = data["visampe"]
    model.updateParList(params)
    vism = model.Amplitude(data["u"], data["v"])
    lnL = -0.5 * ChSq(visa, vism, visae)
    return lnL

#def lnlike_amp(params, data, model):
#    """
#    Calculate the ln likelihood using only the amplitude data.
#    """
#    x, y, ye = data
#    model.updateParList(params)
#    ym = model.Amplitude(*x)
#    lnL = -0.5 * ChSq(y, ym, ye)
#    return lnL

def lnprob_amp(params, data, model):
    """
    Calculate the probability at the parameter spacial position.
    """
    lp = lnprior(params, data, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_amp(params, data, model)

def lnlike_t3(params, data, model):
    """
    Calculate the ln likelihood using both the amplitude and the closure phase.
    """
    visa = data["visamp"]
    visae = data["visampe"]
    t3p = data["t3phi"]
    t3pe = data["t3phie"]
    model.updateParList(params)
    visam = model.Amplitude(data["u"], data["v"])
    t3pm = model.Closure_Phase(data["t3uv1"], data["t3uv2"], data["t3uv3"])
    lnL = -0.5 * ChSq(visa, visam, visae) - 0.5 * ChSq(t3p, t3pm, t3pe)
    return lnL

def lnprob_t3(params, data, model):
    """
    Calculate the probability at the parameter spacial position.
    """
    lp = lnprior(params, data, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_t3(params, data, model)

class EmceeModel(object):
    """
    The MCMC model for emcee.
    """
    def __init__(self, data, model, mode="amplitude"):
        self.__data = data
        self.__model = model
        self.__dim = len(model.get_parVaryList())
        if mode == "amplitude":
            self.__lnprob = lnprob_amp
        elif mode == "closure phase": #Fit the closure phase
            self.__lnprob = lnprob_t3
        else:
            raise ValueError("The mode ({0}) is not recognized!".format(mode))

    def from_prior(self):
        """
        The prior of all the parameters are uniform.
        """
        parList = []
        parDict = self.__model.get_modelParDict()
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                if parFitDict[parName]["vary"]:
                    parRange = parFitDict[parName]["range"]
                    parType  = parFitDict[parName]["type"]
                    if parType == "c":
                        r1, r2 = parRange
                        p = (r2 - r1) * np.random.rand() + r1 #Uniform distribution
                    elif parType == "d":
                        p = np.random.choice(parRange, 1)[0]
                    else:
                        raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                    parList.append(p)
                else:
                    pass
        parList = np.array(parList)
        return parList

    def EnsembleSampler(self, nwalkers, **kwargs):
        """
        Generate the EnsembleSampler.
        """
        self.sampler = emcee.EnsembleSampler(nwalkers, self.__dim, self.__lnprob,
                                             args=[self.__data, self.__model], **kwargs)
        self.__nwalkers = nwalkers
        return self.sampler

    def p_ball(self, p0, ratio=5e-2, nwalkers=None):
        """
        Generate the positions of parameters around the input positions.
        The scipy.stats.truncnorm is used to generate the truncated normal distrubution
        of the parameters within the prior ranges.
        """
        ndim = self.__dim
        if nwalkers is None:
            nwalkers = self.__nwalkers
        pRange = np.array(self.__model.get_parVaryRanges())
        p = np.zeros((nwalkers, ndim))
        for d in range(ndim):
            r0, r1 = pRange[d]
            std = (r1 - r0) * ratio
            loc = p0[d]
            a = (r0 - loc) / std
            b = (r1 - loc) /std
            p[:, d] = truncnorm.rvs(a=a, b=b, loc=loc, scale=std, size=nwalkers)
        return p

    def p_prior(self):
        """
        Generate the positions in the parameter space from the prior.
        The result p0 shape is (nwalkers, dim).
        """
        p0 = [self.from_prior() for i in range(self.__nwalkers)]
        return p0

    def p_logl_max(self, chain=None, lnlike=None, QuietMode=True):
        """
        Find the position in the sampled parameter space that the likelihood is
        the highest.
        """
        if (not chain is None) and (not lnlike is None):
            if not QuietMode:
                print("The chain and lnlike are provided!")
        else:
            chain  = self.sampler.chain
            lnlike = self.sampler.lnprobability
        idx = lnlike.ravel().argmax()
        p   = chain.reshape(-1, self.__dim)[idx]
        return p

    def p_logl_min(self):
        """
        Find the position in the sampled parameter space that the likelihood is
        the lowest.
        """
        chain  = self.sampler.chain
        lnlike = self.sampler.lnprobability
        idx = lnlike.ravel().argmin()
        p   = chain.reshape(-1, self.__dim)[idx]
        return p

    def get_logl(self, p):
        """
        Get the likelihood at the given position.
        """
        return self.__lnprob(p, self.__data, self.__model)

    def run_mcmc(self, pos, iterations, printFrac=1, quiet=False, **kwargs):
        """
        Run the MCMC chain.
        This function just wraps up the sampler.sample() so that there is output
        in the middle of the run.
        """
        if not quiet:
            t0 = time()
        #Notice that the third parameters yielded by EnsembleSampler and PTSampler are different.
        for i, (pos0, lnlike0, logl0) in enumerate(self.sampler.sample(pos, iterations=iterations, **kwargs)):
            if not (i + 1) % int(printFrac * iterations):
                if quiet:
                    pass
                else:
                    progress = 100. * (i + 1) / iterations
                    idx = lnlike0.argmax()
                    lmax = lnlike0[idx]
                    lmin = lnlike0.min()
                    pmax = pos0.reshape((-1, self.__dim))[idx]
                    pname = self.__model.get_parVaryNames(latex=False)
                    print("-----------------------------")
                    print("[{0:<4.1f}%] lnL_max: {1:.3e}, lnL_min: {2:.3e}".format(progress, lmax, lmin))
                    for p, name in enumerate(pname):
                        print("{0:18s} {1:10.3e}".format(name, pmax[p]))
                    print( "**MCMC time elapsed: {0:.3f} min".format( (time()-t0)/60. ) )
        if not quiet:
            print("MCMC finishes!")
        return pos, lnlike0, logl0

    def integrated_time(self):
        """
        Estimate the integrated autocorrelation time of a time series.
        Since it seems there is something wrong with the sampler.integrated_time(),
        I have to calculated myself using acor package.
        """
        chain = self.sampler.chain
        tauParList = []
        for npar in range(self.__dim):
            tauList = []
            for nwal in range(self.__nwalkers):
                pchain = chain[nwal, :, npar]
                try:
                    tau, mean, sigma = acor.acor(pchain)
                except:
                    tau = np.nan
                tauList.append(tau)
            tauParList.append(tauList)
        return tauParList

    def accfrac_mean(self):
        """
        Return the mean acceptance fraction of the sampler.
        """
        return np.mean(self.sampler.acceptance_fraction)

    def posterior_sample(self, burnin=0, fraction=0):
        """
        Return the samples merging from the chains of all the walkers.
        """
        sampler  = self.sampler
        nwalkers = self.__nwalkers
        chain = sampler.chain
        lnprob = sampler.lnprobability[:, -1]
        if burnin > (chain.shape[1]/2.0):
            raise ValueError("The burn-in length ({0}) is too long!".format(burnin))
        if fraction>0:
            lnpLim = np.percentile(lnprob, fraction)
            fltr = lnprob >= lnpLim
            print("ps: {0}/{1} walkers are selected.".format(np.sum(fltr), nwalkers))
            samples = chain[fltr, burnin:, :].reshape((-1, self.__dim))
        else:
            samples = chain[:, burnin:, :].reshape((-1, self.__dim))
        return samples

    def p_median(self, ps=None, **kwargs):
        """
        Return the median value of the parameters according to their posterior
        samples.
        """
        if ps is None:
            ps = self.posterior_sample(**kwargs)
        else:
            pass
        parMedian = np.median(ps, axis=0)
        return parMedian

    def p_uncertainty(self, low=1, center=50, high=99, burnin=50, ps=None, **kwargs):
        """
        Return the uncertainty of each parameter according to its posterior sample.
        """
        if ps is None:
            ps = self.posterior_sample(burnin=burnin, **kwargs)
        else:
            pass
        parRange = np.percentile(ps, [low, center, high], axis=0)
        return parRange

    def print_parameters(self, truths=None, low=1, center=50, high=99, **kwargs):
        """
        Print the ranges of the parameters according to their posterior samples
        and the values of the maximum a posterior (MAP).
        """
        nameList = self.__model.get_parVaryNames(latex=False)
        parRange = self.p_uncertainty(low, center, high, **kwargs)
        pMAP = self.p_logl_max()
        ttList = ["Name", "L ({0}%)".format(low),
                  "C ({0}%)".format(center),
                  "H ({0}%)".format(high), "MAP"]
        if not truths is None:
            ttList.append("Truth")
        tt = " ".join(["{0:12s}".format(i) for i in ttList])
        print("{:-<74}".format(""))
        print(tt)
        for d in range(self.__dim):
            plow = parRange[0, d]
            pcen = parRange[1, d]
            phgh = parRange[2, d]
            pmax = pMAP[d]
            name = nameList[d]
            if (pmax < plow) or (pmax > phgh):
                print("[MCMC Warning]: The best-fit '{0}' is not consistent with its posterior sample".format(name))
            pl = [plow, pcen, phgh]
            info = "{0:12s} {1[0]:<12.3e} {1[1]:<12.3e} {1[2]:<12.3e} {2:<12.3e}".format(name, pl, pmax)
            if truths is None:
                print(info)
            else:
                print(info+" {0:<12.3e}".format(truths[d]))
        p_logl_max = self.p_logl_max()
        print("lnL_max: {0:.3e}".format(self.get_logl(p_logl_max)))

    def Save_Samples(self, filename, **kwargs):
        """
        Save the posterior samples.
        """
        samples = self.posterior_sample(**kwargs)
        np.savetxt(filename, samples, delimiter=",")

    def Save_BestFit(self, filename, low=1, center=50, high=99, **kwargs):
        nameList = self.__model.get_parVaryNames(latex=False)
        parRange = self.p_uncertainty(low, center, high, **kwargs)
        pMAP = self.p_logl_max()
        ttList = ["Name", "L ({0}%)".format(low),
                  "C ({0}%)".format(center),
                  "H ({0}%)".format(high), "MAP"]
        tt = " ".join(["{0:12s}".format(i) for i in ttList])
        fp = open(filename, "w")
        fp.write(tt+"\n")
        for d in range(self.__dim):
            plow = parRange[0, d]
            pcen = parRange[1, d]
            phgh = parRange[2, d]
            pmax = pMAP[d]
            name = nameList[d]
            pl = [plow, pcen, phgh]
            info = "{0:12s} {1[0]:<12.3e} {1[1]:<12.3e} {1[2]:<12.3e} {2:<12.3e}".format(name, pl, pmax)
            fp.write(info+"\n")
        p_logl_max = self.p_logl_max()
        fp.write("#lnL_max: {0:.3e}".format(self.get_logl(p_logl_max)))

    def plot_corner(self, filename=None, burnin=0, fraction=0, ps=None, nuisance=True, **kwargs):
        """
        Plot the corner diagram that illustrate the posterior probability distribution
        of each parameter.
        """
        if ps is None:
            ps = self.posterior_sample(burnin, fraction)
        parname = self.__model.get_parVaryNames()
        dim = self.__dim
        fig = corner.corner(ps[:, 0:dim], labels=parname[0:dim], **kwargs)
        if filename is None:
            return fig
        else:
            plt.savefig(filename)
            plt.close()

    def plot_chain(self, filename=None, truths=None):
        dim = self.__dim
        sampler = self.sampler
        nameList = self.__model.get_parVaryNames()
        chain = sampler.chain
        fig, axes = plt.subplots(dim, 1, sharex=True, figsize=(8, 3*dim))
        for loop in range(dim):
            axes[loop].plot(chain[:, :, loop].T, color="k", alpha=0.4)
            axes[loop].yaxis.set_major_locator(MaxNLocator(5))
            if not truths is None:
                axes[loop].axhline(truths[loop], color="r", lw=2)
            axes[loop].set_ylabel(nameList[loop], fontsize=24)
        if filename is None:
            return (fig, axes)
        else:
            plt.savefig(filename)
            plt.close()

    def plot_lnlike(self, filename=None, iterList=[0.5, 0.8, 1.0], **kwargs):
        lnprob = self.sampler.lnprobability
        _, niter = lnprob.shape
        iterList = np.around(niter * np.array(iterList)) - 1
        fig = plt.figure()
        for i in iterList:
            l = lnprob[:, int(i)]
            plt.hist(l[~np.isinf(l)], label="iter: {0}".format(i), **kwargs)
        plt.legend(loc="upper left")
        if filename is None:
            ax = plt.gca()
            return (fig, ax)
        else:
            plt.savefig(filename)
            plt.close()

    def reset(self):
        """
        Reset the sampler, for completeness.
        """
        self.sampler.reset()

    def diagnose(self):
        """
        Diagnose whether the MCMC run is reliable.
        """
        nameList = self.__model.get_parVaryNames(latex=False)
        print("---------------------------------")
        print("Mean acceptance fraction: {0:.3f}".format(self.accfrac_mean()))
        print("PN       : ACT (min-max)")
        it = self.integrated_time()
        for loop in range(self.__dim):
            itPar = it[loop]
            print("{0:9s}: {i[0]:.3f}-{i[1]:.3f}".format(nameList[loop], i=[min(itPar), max(itPar)]))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

    def __del__(self):
        del self.__data
        del self.__model
        parList = self.__dict__.keys()
        if "sampler" in parList:
            del self.sampler
