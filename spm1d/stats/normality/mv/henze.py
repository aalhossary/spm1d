
'''
Multivariate normality testing (Henze-Zirkler test)

All procedures fit the given model, calculate residuals then use
"normality.residuals" to conduct a Henze-Zirkler test
on the multivariate residuals.
'''



from math import sqrt,exp,log
import numpy as np
from scipy import stats
from .... import rft1d

from ... _spm import SPM0D_Z, SPM_Z
from ... _mvbase import _fwhm as estimate_mvfwhm
from ... _mvbase import _resel_counts as mv_resel_counts



def henze_zirkler_single_node(y):
	X    = np.matrix( y.copy() )
	n,p  = X.shape
	n,p  = float(n), float(p)
	### shape information:
	b    = 2**-0.5 * ((2*p + 1)/4)**(1/(p + 4))*(n**(1/(p + 4)))
	b2   = b**2
	### covariance:
	S     = np.matrix(np.cov(y.T, ddof=0))
	difT  = np.matrix(X - X.mean(axis=0))
	Si    = np.linalg.inv(S)
	Dj    = np.diag( difT * Si * difT.T  )
	Y     = X * Si * X.T
	diagY = np.matrix( np.diag(Y) ).T
	ones  = np.matrix( np.ones(int(n)) ).T
	Djk   = -2*Y.T + (diagY * ones.T) + (ones * diagY.T)
	### test statistic:
	if np.linalg.matrix_rank(X) == p:
		a0  = np.exp( -b2/2 * Djk ).sum()
		a1  = (1 + b2)**(-p/2)
		a2  = np.exp( (-b2 * Dj) / (2 + 2*b2) ).sum()
		a3  = (1 + 2*b2)**(-p/2)
		HZ  = n * (a0 * n**-2  - (2*a1*a2/n)  + a3)
	else:
		HZ  = n * 4
	return HZ
	# ### shape parameters:
	# wb   = (1 + b2) * (1 + 3*b2)
	# a    = 1 + 2*b2;
	# mu   = 1 - a**(-p/2) *   (1 + p*b2/a + (p*(p + 2)*(b2*b2))/(2*a**2))
	# mu2  = mu**2
	# si2  = 2*(1 + 4*b2)**(- p/2) + 2*a**(-p)*(1 + (2*p*b2*b2)/a**2 + (3*p *
	#     (p + 2)*b**8)/(4*a**4)) - 4*wb**( - p/2)*(1 + (3*p*b**4)/(2*wb) + (p *
	#     (p + 2)*b**8)/(2*wb**2))
	# pmu = log(   sqrt(  mu2*mu2 / (si2 + mu2)  )   )
	# psi = sqrt(  log(  (si2 + mu2)/mu2  )  )
	# p   = lognorm_sf(HZ, pmu, psi)
	# return HZ #, pmu, psi, p




def residuals(y):
	'''
	Compute the Mardia-Rencher test statistic for skewness from a set of model residuals.
	'''
	y           = np.asarray(y)
	J,I         = y.shape[0], y.shape[-1]
	mu,sigma    = rft1d.util.henze_zirkler_mu_sigma(J, I)
	# if J < 20:
	# 	raise( ValueError('In order to conduct multivariate normality tests there must at least 20 observations. Only %d found.' %J)   )
	if np.ndim(y)==2:
		z0      = henze_zirkler_single_node(y)
		z       = rft1d.util.lognorm2norm(z0, mu, sigma)
		spm     = SPM0D_Z(z, residuals=y)   #skewness
	else:
		Q       = y.shape[1]
		z0      = np.array( [henze_zirkler_single_node(y[:,i,:])   for i in range(Q)] ).T
		z       = rft1d.util.lognorm2norm(z0, mu, sigma)
		fwhm    = estimate_mvfwhm(y)
		resels  = mv_resel_counts(y, fwhm, roi=None)
		spm     = SPM_Z(z, fwhm, resels, residuals=y)
	spm.z0     = z0
	return spm




def onesample(y):
	r   = y - y.mean(axis=0)
	return residuals(r)




# def residuals()



# def henze_zirkler_1d(y):
# 	n,Q,p  = y.shape
# 	n,p    = float(n), float(p)
# 	### shape information:
# 	b    = 2**-0.5 * ((2*p + 1)/4)**(1/(p + 4))*(n**(1/(p + 4)))
# 	b2   = b**2
# 	### covariance:
# 	HZ   = []
# 	for i in range(Q):
# 		X     = np.matrix( y[:,i,:] )
# 		S     = np.matrix( np.cov(X.T, ddof=0) )
# 		difT  = np.matrix(X - X.mean(axis=0))
# 		Si    = np.linalg.inv(S)
# 		Dj    = np.diag( difT * Si * difT.T  )
# 		Y     = X * Si * X.T
# 		diagY = np.matrix( np.diag(Y) ).T
# 		ones  = np.matrix( np.ones(int(n)) ).T
# 		Djk   = -2*Y.T + (diagY * ones.T) + (ones * diagY.T)
# 		### test statistic:
# 		if np.linalg.matrix_rank(X) == p:
# 			a0  = np.exp( -b2/2 * Djk ).sum()
# 			a1  = (1 + b2)**(-p/2)
# 			a2  = np.exp( (-b2 * Dj) / (2 + 2*b2) ).sum()
# 			a3  = (1 + 2*b2)**(-p/2)
# 			hz  = n * (a0 * n**-2  - (2*a1*a2/n)  + a3)
# 		else:
# 			hz  = n * 4
# 		HZ.append(hz)
# 	return np.array(HZ)



