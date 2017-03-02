

'''
Multivariate normality testing (Roytson's test)

All procedures fit the given model, calculate residuals then use
"normality.residuals" to conduct Roytson's test
on the multivariate residuals.
'''

# Copyright (C) 2016  Todd Pataky

from math import log,exp
import numpy as np
import scipy.stats
from ... _spm import SPM0D_X2, SPM_X2
from ... _mvbase import _fwhm as estimate_mvfwhm
from ... _mvbase import _resel_counts as mv_resel_counts
from .... import rft1d




def shapiro_wilks(x):
	x      = np.sort(x)
	kurt   = scipy.stats.kurtosis(x, fisher=False)
	if kurt > 3:
		#use Shapiro-Francia stat (better for leptokurtic data)
		n  = x.size
		xx = (np.arange(1,n+1)-3./8) / (n+0.25)
		m  = scipy.stats.norm.isf(  1 - xx    )
		w  = 1.0 / np.linalg.norm( m ) * m
		d  = x - x.mean()
		W  = np.dot(w, x)**2   /  np.dot(d, d)
	else:
		#use Shapiro-Wilks stat
		W  = scipy.stats.shapiro(x)[0]  #slightly different from R's results
	return W



def royston_single_node(X):
	'''
	Modified from the MATLAB file:
	"ROYSTEST_Royston's_Multivariate_Normality_Test.m"
	available here:
	# https://www.researchgate.net/publication/255982178_ROYSTEST_Royston%27s_Multivariate_Normality_Test
	'''
	n,p = X.shape
	p   = float(p)
	x   = log(n)
	g   = 0
	m   = -1.5861 - 0.31082*x - 0.083751*x**2 + 0.0038915*x**3
	s   = exp(-0.4803 -0.082676*x + 0.0030302*x**2)
	W   = np.array([shapiro_wilks(xx)  for xx in X.T])
	Z   = ( np.log(1 - W) + g - m) / s
	R   = scipy.stats.norm.isf(  scipy.stats.norm.cdf(-Z)/2  )**2
	u   = 0.715
	v   = 0.21364 + 0.015124*x**2 - 0.0018034*x**3
	l   = 5
	C   = np.corrcoef(X.T)
	NC  = (C**l) * (1 - (u*(1 - C)**u) / v )
	T   = NC.sum() - p
	mC  = T / (p*p - p)
	e   = p / (1 + (p-1)*mC)
	H   = e * R.sum() / p  #Royston's statistic
	return H,e


def residuals(y):
	'''
	Compute the Mardia-Rencher test statistics for skewness and kurtosis test statistics
	for a set of model residuals.
	'''
	y     = np.asarray(y)
	J,I   = y.shape[0], y.shape[-1]
	# if J < 20:
	# 	raise( ValueError('In order to conduct multivariate normality tests there must at least 20 observations. Only %d found.' %J)   )
	if np.ndim(y)==2:
		w,e    = royston_single_node(y)
		spm    = SPM0D_X2(w, (1,e), residuals=y)
	else:
		Q      = y.shape[1]
		w,e    = np.array([royston_single_node(y[:,i,:])  for i in range(Q)]).T
		v      = e.mean()
		fwhm   = estimate_mvfwhm(y)
		resels = mv_resel_counts(y, fwhm, roi=None)
		spm    = SPM_X2(w, (1,v), fwhm, resels, residuals=y)
	return spm




def onesample(y):
	r   = y - y.mean(axis=0)
	return residuals(r)
	
