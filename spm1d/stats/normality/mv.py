
'''
Multivariate normality testing

All procedures fit the given model, calculate residuals then use
"normality.residuals" to conduct a Mardia-Rencher test
on the multivariate residuals.
'''

# Copyright (C) 2016  Todd Pataky

import warnings
from math import log
import numpy as np
from .. _spm import SPM0D_X2, SPM_X2, SPM0D_Z, SPM_Z
from .. _mvbase import _fwhm as estimate_mvfwhm
from .. _mvbase import _resel_counts as mv_resel_counts
from .. t import regress as _main_regress
from .. anova import anova1 as _main_anova1
from .. anova import anova1rm as _main_anova1rm
from .. anova import anova2 as _main_anova2
from .. anova import anova2nested as _main_anova2nested
from .. anova import anova2rm as _main_anova2rm
from .. anova import anova2onerm as _main_anova2onerm
from .. anova import anova3 as _main_anova3
from .. anova import anova3nested as _main_anova3nested
from .. anova import anova3rm as _main_anova3rm
from .. anova import anova3onerm as _main_anova3onerm
from .. anova import anova3tworm as _main_anova3tworm
from ... import rft1d



def _assemble_results(a, b):
	if a is None:
		result = b
	elif b is None:
		result = a
	else:
		result = a,b
	return result



def mr_single_node(y, skewness=True, kurtosis=False):
	'''
	Compute the Mardia-Rencher test statistics for skewness and kurtosis at a single node.
	
	References:
	
	Mardia KV (1970). Measures of multivariate skewness and kurtosis with applications.
	Biometrika, 57(3), 519--530. http://doi.org/10.2307/2334770
	
	Rencher AC and Christensen WF (2012). Methods of Multivariate Analysis, 3rd Edition.
	New York: Wiley, pp. 107--108.
	
	See also:
	https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Multivariate_normality_tests
	'''
	### check whether skewness or kurtosis should be calculated:
	skewness = bool(skewness)
	kurtosis = bool(kurtosis)
	if (not skewness) and (not kurtosis):
		raise( ValueError('At least one of the "kurtosis" and "skewness" keyword arguments must be True.')   )
	### assemble array shape information:
	J,I      = y.shape   #nResponses, nVector components
	J,I      = float(J), float(I)
	### compute SIGMA:
	m        = y.mean(axis=0)
	D        = np.matrix(y - m)
	SIGMA    = np.cov( D.T, ddof=0)
	SIGMAi   = np.matrix( np.linalg.inv( SIGMA ) )
	a,b      = None, None
	### compute skewness and/or kurtosis:
	zsum     = np.array( D * SIGMAi * D.T )
	if skewness:
		### The calculation from Mardia (1970) is:
		### z       = (z**3).sum() / (6.0 * J)
		### A better calculation (especially for small samples)
		### is from Rencher & Christensen (2012, p.108) is:
		z       = (zsum**3).sum() / (J**2)
		z0      = (I+1) * (J+1) * (J+3)
		z1      = 6 * ( (I+1) * (J+1) - 6 )
		a       = (z0 / z1) * z    #skewness statistic
	if kurtosis:
		### compute kurtosis:
		b       = (np.diag(zsum)**2).sum() / J
		if J >= 500:   #Rencher & Christensen (2012, p.108, eqn. 4.39)
			b  -= I * (I + 2)
			b  *= (   J / ( 8 * I * (I+2) )   )**0.5
		else:   #Rencher & Christensen (2012, p.108, Eqn. 4.40)
			### NOTE!!!   Eqn.4.40 uses (J + I + 1) but (J - I - 1) gives better results
			b  -= I * (I + 2) * (J - I - 1) / J
			b  *= (   (J-1) / ( 8 * I * (I+2)  )   )**0.5
	#return results:
	return a,b
	# return _assemble_results(a, b)
	# if skewness and kurtosis:
	# 	result = a,b
	# elif skewness:
	# 	result = a
	# else:
	# 	result = b
	# return result


def residuals(y, skewness=True, kurtosis=False):
	'''
	Compute the Mardia-Rencher test statistics for skewness and kurtosis test statistics
	for a set of model residuals.
	'''
	y     = np.asarray(y)
	J,I   = y.shape[0], y.shape[-1]
	if J < 20:
		raise( ValueError('In order to conduct multivariate normality tests there must at least 20 observations. Only %d found.' %J)   )
	if J < 50:
		s = '\n\nSmall sample size (N=%d).  Multivariate tests for small sample sizes (N < 50) may not accurate.  Interpret results cautiously.\n' %J
		warnings.warn(s, UserWarning, stacklevel=2)
	v     = 1./6 * I * (I+1) * (I+2)   #degrees of freedom (for skewness)
	if np.ndim(y)==2:
		z0,z1     = mr_single_node(y, skewness=skewness, kurtosis=kurtosis)
		### build SPM object(s):
		spm0,spm1 = None,None
		if z0 is not None:
			spm0  = SPM0D_X2(z0, (1,v), residuals=y)   #skewness
		if z1 is not None:
			spm1  = SPM0D_Z(z1, residuals=y)           #kurtosis
	else:
		Q         = y.shape[1]
		z0,z1     = np.array( [mr_single_node(y[:,i,:], skewness, kurtosis)   for i in range(Q)] ).T
		fwhm      = estimate_mvfwhm(y)
		resels    = mv_resel_counts(y, fwhm, roi=None)
		### build SPM object(s):
		spm0,spm1 = None,None
		if skewness:
			spm0  = SPM_X2(z0, (1,v), fwhm, resels, residuals=y)   #skewness
		if kurtosis:
			spm1  = SPM_Z(z1, fwhm, resels, residuals=y)         #kurtosis
		
	return _assemble_results(spm0, spm1)




# def _stack_data(*args):
# 	return np.hstack(args) if (np.ndim(args[0])==1) else np.vstack(args)
#
def hotellings(y, skewness=True, kurtosis=False):
	r   = y - y.mean(axis=0)
	return residuals(r, skewness, kurtosis)

# def ttest_paired(yA, yB):
# 	return ttest( yA - yB )
#
# def ttest2(yA, yB):
# 	rA   = yA - yA.mean(axis=0)
# 	rB   = yB - yB.mean(axis=0)
# 	r    = _stack_data(rA, rB)
# 	return residuals(r)







