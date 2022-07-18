import pytest
import unittest
import numpy as np 
import scipy.stats as stats
from ..src.gen_gamma import GeneralizedGamma


class TestGeneralizedGamma(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGeneralizedGamma, self).__init__(*args, **kwargs)
        self.N = 1_000_000
        self.alpha = 0.01 # alpha of hypothesis test. 
    
    def test_exponential_pdf_matches(self):
        '''If  𝛼=𝜌=1, then 𝐺𝐺(1,1,𝜆)∼Exponential(𝜆)'''
        x = np.arange(1, 11)
        a = p = 1
        lambd = 5
        
        GG = GeneralizedGamma(a, p, lambd)
        expon_rv = stats.expon(scale=lambd)
        true_pdf = expon_rv.pdf(x)
        assert np.allclose(GG.pdf(x), true_pdf)
    
    def test_exponential_cdf_matches(self):
        '''If  𝛼=𝜌=1, then 𝐺𝐺(1,1,𝜆)∼Exponential(𝜆)'''
        x = np.arange(1, 11)
        a = p = 1
        lambd = 5
        
        GG = GeneralizedGamma(a, p, lambd)
        expon_rv = stats.expon(scale=lambd)
        true_cdf = expon_rv.cdf(x)
        assert np.allclose(GG.cdf(x), true_cdf)

    def test_gamma_pdf_matches(self):
        '''If  𝜌=1, then 𝐺𝐺(𝛼,1,𝜆)∼Gamma(𝛼,𝜆)'''
        x = np.arange(1, 11)
        a, p, lambd = 2,1,5
        
        GG = GeneralizedGamma(a, p, lambd)
        gamma_rv = stats.gamma(a, scale=lambd)
        true_pdf = gamma_rv.pdf(x)
        assert np.allclose(GG.pdf(x), true_pdf)

    def test_gamma_cdf_matches(self):
        '''If  𝜌=1, then 𝐺𝐺(𝛼,1,𝜆)∼Gamma(𝛼,𝜆)'''
        x = np.arange(1, 11)
        a, p, lambd = 2,1,5
        
        GG = GeneralizedGamma(a, p, lambd)
        gamma_rv = stats.gamma(a, scale=lambd)
        true_cdf = gamma_rv.cdf(x)
        assert np.allclose(GG.cdf(x), true_cdf)

    def test_weibull_pdf_matches(self):
        '''If  𝛼=𝜌, then 𝐺𝐺(𝜌,𝜌,𝜆)∼W(𝜌,𝜆)'''
        x = np.arange(1, 11)
        a = p = 10
        lambd = 5
        
        GG = GeneralizedGamma(a, p, lambd)
        weibull_rv = stats.weibull_min(p, scale=lambd)
        true_pdf = weibull_rv.pdf(x)
        assert np.allclose(GG.pdf(x), true_pdf)
        
    def test_weibull_cdf_matches(self):
        '''If  𝛼=𝜌, then 𝐺𝐺(𝜌,𝜌,𝜆)∼W(𝜌,𝜆)'''
        x = np.arange(1, 11)
        a = p = 10
        lambd = 5
        
        GG = GeneralizedGamma(a, p, lambd)
        weibull_rv = stats.weibull_min(p, scale=lambd)
        true_cdf = weibull_rv.cdf(x)
        assert np.allclose(GG.cdf(x), true_cdf)

    def test_exponential_ppf_matches(self):
        '''If  𝛼=𝜌=1, then 𝐺𝐺(1,1,𝜆)∼Exponential(𝜆)'''
        q = np.arange(0, 1+0.001, 0.1)
        a = p = 1
        lambd = 5

        GG = GeneralizedGamma(a, p, lambd)
        expon_rv = stats.expon(scale=lambd)
        true_ppf = expon_rv.ppf(q)
        assert np.allclose(GG.ppf(q), true_ppf)
    
    def test_gamma_ppf_matches(self):
        '''If  𝜌=1, then 𝐺𝐺(𝛼,1,𝜆)∼Gamma(𝛼,𝜆)'''
        q = np.arange(0, 1+0.001, 0.1)
        a, p, lambd = 2,1,5

        GG = GeneralizedGamma(a, p, lambd)
        gamma_rv = stats.gamma(a, scale=lambd)
        true_ppf = gamma_rv.ppf(q)
        assert np.allclose(GG.ppf(q), true_ppf)
    
    def test_weibull_ppf_matches(self):
        '''If  𝛼=𝜌, then 𝐺𝐺(𝜌,𝜌,𝜆)∼W(𝜌,𝜆)'''
        q = np.arange(0, 1+0.001, 0.1)
        a = p = 10
        lambd = 5

        GG = GeneralizedGamma(a, p, lambd)
        weibull_rv = stats.weibull_min(p, scale=lambd)
        true_ppf = weibull_rv.ppf(q)
        assert np.allclose(GG.ppf(q), true_ppf)

    def test_exponential_sampling_matches(self):
        '''If  𝛼=𝜌=1, then 𝐺𝐺(1,1,𝜆)∼Exponential(𝜆)'''
        a = p = 1
        lambd = 5

        GG = GeneralizedGamma(a, p, lambd)
        expon_rv = stats.expon(scale=lambd)
        
        sample = expon_rv.rvs(self.N)
        _, pval = stats.ks_2samp(GG.sample(size=self.N), sample)
        assert pval > self.alpha
        
    def test_gamma_sampling_matches(self):
        '''If  𝜌=1, then 𝐺𝐺(𝛼,1,𝜆)∼Gamma(𝛼,𝜆)'''
        q = np.arange(0, 1+0.001, 0.1)
        a, p, lambd = 2,1,5

        GG = GeneralizedGamma(a, p, lambd)
        gamma_rv = stats.gamma(a, scale=lambd)
        
        sample = gamma_rv.rvs(self.N)
        _, pval = stats.ks_2samp(GG.sample(size=self.N), sample)
        assert pval > self.alpha

    def test_weibull_sampling_matches(self):
        '''If  𝛼=𝜌, then 𝐺𝐺(𝜌,𝜌,𝜆)∼W(𝜌,𝜆)'''
        a = p = 10
        lambd = 5

        GG = GeneralizedGamma(a, p, lambd)
        weibull_rv = stats.weibull_min(p, scale=lambd)
        
        sample = weibull_rv.rvs(self.N)
        _, pval = stats.ks_2samp(GG.sample(size=self.N), sample)
        assert pval > self.alpha
    
    def test_sample(self):
        '''Compares the eCDF from randomly drawn samples to the cdf'''
        a, p, lambd = 2,3,5
        GG = GeneralizedGamma(a, p, lambd)
        _, pval = stats.kstest(GG.sample(size=self.N), GG.cdf)
        assert pval > self.alpha

    def test_fit(self):
        '''Makes sure that the fit method returns the expected parameter values'''
        theta = np.array([5,11,60])

        y = GeneralizedGamma(*theta).sample(self.N)
        GG = GeneralizedGamma.fit(y)
        theta_hat = np.array([GG.α, GG.p, GG.λ])
        assert np.allclose(theta, theta_hat, atol=1e-1)
