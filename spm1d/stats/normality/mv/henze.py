
# from math import sqrt,log,exp,erf
from math import sqrt,exp,log
import numpy as np
from scipy import stats

# def lognorm_sf(x, mu=0, sigma=1):
#    a = (log(x) - mu) / sqrt(2*sigma**2)
#    p = 0.5 + 0.5 * erf(a)
#    return 1 - p


def lognorm_sf(x, mu, sigma):
	'''
	http://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma/42481670#42481670
	'''
	shape,loc,scale  = sigma, 0, exp(mu)
	return stats.lognorm.sf(x, shape, loc, scale)

def lognorm_isf(x, mu, sigma):
	shape,loc,scale  = sigma, 0, exp(mu)
	return stats.lognorm.isf(x, shape, loc, scale)



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
	### shape parameters:
	wb   = (1 + b2) * (1 + 3*b2)
	a    = 1 + 2*b2;
	mu   = 1 - a**(-p/2) *   (1 + p*b2/a + (p*(p + 2)*(b2*b2))/(2*a**2))
	mu2  = mu**2
	si2  = 2*(1 + 4*b2)**(- p/2) + 2*a**(-p)*(1 + (2*p*b2*b2)/a**2 + (3*p *
	    (p + 2)*b**8)/(4*a**4)) - 4*wb**( - p/2)*(1 + (3*p*b**4)/(2*wb) + (p *
	    (p + 2)*b**8)/(2*wb**2))
	pmu = log(   sqrt(  mu2*mu2 / (si2 + mu2)  )   )
	psi = sqrt(  log(  (si2 + mu2)/mu2  )  )
	p   = lognorm_sf(HZ, pmu, psi)
	return HZ, pmu, psi, p


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



