
'''
Stand-alone utilities including variable and distribution transformations.
'''

# Copyright (C) 2017  Todd Pataky



from math import log,sqrt,exp
import numpy as np



def henze_zirkler_mu_sigma(J, I):
	'''
	J = number of observations
	I = number of vector components

	Modified from Antonio Trujillo-Ortiz (2009):
	https://www.mathworks.com/matlabcentral/fileexchange/17931-hzmvntest
	Thank you!!
	'''
	n,p    = float(J), float(I)
	### shape information:
	b      = 2**-0.5 * ((2*p + 1)/4)**(1/(p + 4))*(n**(1/(p + 4)))
	b2     = b**2
	### shape parameters:
	wb     = (1 + b2) * (1 + 3*b2)
	a      = 1 + 2*b2
	mu     = 1 - a**(-p/2) *   (1 + p*b2/a + (p*(p + 2)*(b2*b2))/(2*a**2))
	mu2    = mu**2
	si2    = 2*(1 + 4*b2)**(- p/2) + 2*a**(-p)*(1 + (2*p*b2*b2)/a**2 + (3*p *
	    (p + 2)*b**8)/(4*a**4)) - 4*wb**( - p/2)*(1 + (3*p*b**4)/(2*wb) + (p *
	    (p + 2)*b**8)/(2*wb**2))
	### final mu and sigma:
	pmu    = log(   sqrt(  mu2*mu2 / (si2 + mu2)  )   )
	psigma = sqrt(  log(  (si2 + mu2)/mu2  )  )
	return pmu, psigma



def lognorm2norm(x, mu, sigma):
	'''
	Transform log-normal values into standard normal values
	'''
	return (np.log(x) - mu) / sigma


def norm2lognorm(z, mu, sigma):
	'''
	Transform standard normal values into log-normal values
	'''
	return np.exp(mu + sigma*z)





