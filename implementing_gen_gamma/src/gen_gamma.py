import numpy as np
import scipy.stats as stats
import scipy.special as sp
from scipy.optimize import minimize 
import numpy.typing as npt
from typing import Tuple, Optional, Union


# Type hint shortcut: tells us its a numpy array of float types
FloatArray = npt.NDArray[np.float64]
RandomState = np.random.RandomState 

class GeneralizedGamma:
    '''The generalized gamma distribution.'''

    def __init__(self, α: float, p: float, λ: float):
        '''A generalized gamma distribution.
        '''        
        self.α = α
        self.p = p
        self.λ = λ
        
    def __repr__(self):
        '''Specifies what text a print statement should return'''
        return f'GG(α={round(self.α,2)}, p={round(self.p,2)}, λ={round(self.λ,2)})'
    
    def expectation(self) -> FloatArray:
        '''Returns the expected value of the distribution'''
        α, p, λ = self.α, self.p, self.λ
        return λ * sp.gamma( (α+1)/p ) / sp.gamma(α/p)
        
    def pdf(self, x: Union[float, FloatArray]) -> FloatArray:
        '''The probability density function evaluated at x
        
        Parameters
        -----------
            x: A float or an array of floats to evaluate the probability density
        
        Returns
        --------
            the probability density function evaluated at x
        '''        
        α, p, λ = self.α, self.p, self.λ
        return ((p/λ)*(x/λ)**(α-1)* np.exp(-1*(x/λ)**p)) / sp.gamma(α/p) 
        
    def cdf(self, x: Union[float, FloatArray]) -> FloatArray:
        '''The cumulative density function evaluated at x
        
        Parameters
        -----------
            x: A float or an array of floats to evaluate the cumulative probability density
        
        Returns
        --------
            the cumulative density function evaluated at x
        
        '''       
        α, p, λ = self.α, self.p, self.λ
        return sp.gammainc(α/p, (x/λ)**p)
    
    def sample(self, size: int = 1, seed: Optional[int] = None, rng: Optional[int] = None) -> FloatArray:
        '''Draws samples using the random inversion method
        
        Parameters
        -----------
            size: The number of samples to draw from the distribution
            seed: A random seed to use for sampling
            rng: A random state to use for sampling. Overwrites seed. Required for Aesara
        
        Returns
        --------
            An array of randomly sampled draws from the distribution
        '''
        if seed:
            np.random.seed(seed)

        if rng:
            q = rng.uniform(size=size)
        else:
            q = np.random.uniform(size=size)
        return self.ppf(q) 
    
    def ppf(self, q: float) -> FloatArray:
        '''The percentile point function, or quantile function,
        of the Generalized Gamma
        
        Parameters
        -----------
            q: A quantile from 0-1
        
        Returns
        --------
            The quantile function evaluated at q
        '''
        α, p, λ = self.α, self.p, self.λ
        return λ * stats.gamma(a=α/p, scale=1).ppf(q) **(1/p)

    def logp(self, x: Union[float, FloatArray]) -> FloatArray:
        '''The log probability density function evaluated at x
        
        Parameters
        -----------
            x: A float or an array of floats to evaluate the log probability density
        
        Returns
        --------
            the log probability density function evaluated at x
        '''        
        α, p, λ = self.α, self.p, self.λ
        return (
            np.log(p) - np.log(λ)
            + (α-1)*np.log(x/λ)
            - (x/λ)**p
            - sp.loggamma(α/p)
        )

    def fit(y):
        
        def _negative_log_likelihood(log_theta, y):
            params = np.exp(log_theta)
            dist = GeneralizedGamma( *params )
            LL = dist.logp(y)
            return (-1 * LL).sum()

        mle_estimate = mle_estimate = minimize(
           _negative_log_likelihood, 
            x0=np.log(np.array([1,1,10])), 
            args=(y,), 
            method='L-BFGS-B')

        theta_hat = np.exp( mle_estimate.x )
        return GeneralizedGamma(*theta_hat)

