
'''
Multivariate normality testing

All procedures fit the given model, calculate residuals then use
"normality.residuals" to conduct a Mardia-Rencher test
on the multivariate residuals.
'''

# Copyright (C) 2017  Todd Pataky

import numpy as np
from ... _spm import SPM0D_X2, SPM_X2
from ... _mvbase import _fwhm as estimate_mvfwhm
from ... _mvbase import _resel_counts as mv_resel_counts




def skewness_single_node(y):
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
	### assemble array shape information:
	J,I      = map(float, y.shape)   #nResponses, nVector components
	### compute SIGMA:
	m        = y.mean(axis=0)
	D        = np.matrix(y - m)
	SIGMA    = np.cov( D.T, ddof=0)
	SIGMAi   = np.matrix( np.linalg.inv( SIGMA ) )
	a,b      = None, None
	### compute skewness:
	mij      = np.array( D * SIGMAi * D.T )
	skew     = (mij**3).sum() / (J**2)
	### test statistic:
	z0       = (I+1) * (J+1) * (J+3)
	z1       = 6 * (  (I+1) * (J+1) - 6 )
	chi2     = (z0 / z1) * skew
	return chi2




def residuals(y):
	'''
	Compute the Mardia-Rencher test statistic for skewness from a set of model residuals.
	'''
	y     = np.asarray(y)
	J,I   = y.shape[0], y.shape[-1]
	if J < 20:
		raise( ValueError('In order to conduct multivariate normality tests there must at least 20 observations. Only %d found.' %J)   )
	v     = 1./6 * I * (I+1) * (I+2)   #degrees of freedom (for skewness)
	if np.ndim(y)==2:
		z      = skewness_single_node(y)
		spm    = SPM0D_X2(z, (1,v), residuals=y)   #skewness
	else:
		Q         = y.shape[1]
		z         = np.array( [skewness_single_node(y[:,i,:])   for i in range(Q)] ).T
		fwhm      = estimate_mvfwhm(y)
		resels    = mv_resel_counts(y, fwhm, roi=None)
		spm       = SPM_X2(z, (1,v), fwhm, resels, residuals=y)
	return spm








# def mardia_single_node(y, skewness=True, kurtosis=False):
# 	# return None
# 	'''
# 	Compute the Mardia-Rencher test statistics for skewness and kurtosis at a single node.
#
# 	References:
#
# 	Mardia KV (1970). Measures of multivariate skewness and kurtosis with applications.
# 	Biometrika, 57(3), 519--530. http://doi.org/10.2307/2334770
#
# 	Rencher AC and Christensen WF (2012). Methods of Multivariate Analysis, 3rd Edition.
# 	New York: Wiley, pp. 107--108.
#
# 	See also:
# 	https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Multivariate_normality_tests
# 	'''
# 	### check whether skewness or kurtosis should be calculated:
# 	skewness = bool(skewness)
# 	kurtosis = bool(kurtosis)
# 	if (not skewness) and (not kurtosis):
# 		raise( ValueError('At least one of the "kurtosis" and "skewness" keyword arguments must be True.')   )
# 	### assemble array shape information:
# 	J,I      = y.shape   #nResponses, nVector components
# 	J,I      = float(J), float(I)
# 	### compute SIGMA:
# 	m        = y.mean(axis=0)
# 	D        = np.matrix(y - m)
# 	SIGMA    = np.cov( D.T, ddof=0)
# 	SIGMAi   = np.matrix( np.linalg.inv( SIGMA ) )
# 	a,b      = None, None
# 	### compute skewness and/or kurtosis:
# 	mij      = np.array( D * SIGMAi * D.T )
# 	if skewness:
# 		skew = (mij**3).sum() / (J**2)
# 		if J > 500:
# 			### from Mardia (1970):
# 			# skew  = (zsum**3).sum() / (6 * J)
#
# 		else:
# 			### A better calculation (especially for small samples)
# 			### is from Rencher & Christensen (2012, p.108) is:
# 			skew  = (mij**3).sum() / (J**2)
# 			z0    = (I+1) * (J+1) * (J+3)
# 			z1    = 6 * (  (I+1) * (J+1) - 6 )
# 			chi2  = (z0 / z1) * skew  #test statistic (~Chi2 distributed)
#
# 	if kurtosis:
# 		### compute kurtosis:
# 		kurt      = (np.diag(mij)**2).sum() / J
# 		if J > 500:
# 			z     = (kurt - I * (I + 2)) * sqrt( J / ( 8 * I * (I+2) ) )
# 		else:
# 			z0    = I * (I + 2) * (J - I - 1) / J
# 			z1    = sqrt( (J-1) / ( 8 * I * (I+2) )  )
# 			z     = (kurt - z0) * z1
#
#
#
#
# 		#Rencher & Christensen (2012, p.108, eqn. 4.39)
#
#
#
# 		# if J >= 500:   #Rencher & Christensen (2012, p.108, eqn. 4.39)
# 		# 	b  -= I * (I + 2)
# 		# 	b  *= (   J / ( 8 * I * (I+2) )   )**0.5
# 		# else:   #Rencher & Christensen (2012, p.108, Eqn. 4.40)
# 		# 	### NOTE!!!   Eqn.4.40 uses (J + I + 1) but (J - I - 1) gives better results
# 		# 	# b  -= I * (I + 2) * (J - I - 1) / J
# 		# 	b  -= I * (I + 2) * (J + I + 1) / J
# 		# 	b  *= (   (J-1) / ( 8 * I * (I+2)  )   )**0.5
# 		# print(b0, b)
# 	#return results:
# 	return skew,kurt,chi2,z
# 	# return _assemble_results(a, b)
# 	# if skewness and kurtosis:
# 	# 	result = a,b
# 	# elif skewness:
# 	# 	result = a
# 	# else:
# 	# 	result = b
# 	# return result




# def residuals(y, skewness=True, kurtosis=False):
# 	'''
# 	Compute the Mardia-Rencher test statistics for skewness and kurtosis test statistics
# 	for a set of model residuals.
# 	'''
# 	y     = np.asarray(y)
# 	J,I   = y.shape[0], y.shape[-1]
# 	# if J < 20:
# 	# 	raise( ValueError('In order to conduct multivariate normality tests there must at least 20 observations. Only %d found.' %J)   )
# 	if J < 50:
# 		s = '\n\nSmall sample size (N=%d).  Multivariate tests for small sample sizes (N < 50) may not accurate.  Interpret results cautiously.\n' %J
# 		warnings.warn(s, UserWarning, stacklevel=2)
# 	v     = 1./6 * I * (I+1) * (I+2)   #degrees of freedom (for skewness)
# 	if np.ndim(y)==2:
# 		s,k,z0,z1 = mr_single_node(y, skewness=skewness, kurtosis=kurtosis)
# 		### build SPM object(s):
# 		spm0,spm1 = None,None
# 		if z0 is not None:
# 			spm0  = SPM0D_X2(z0, (1,v), residuals=y)   #skewness
# 			spm0.skewness = s
# 		if z1 is not None:
# 			spm1  = SPM0D_Z(z1, residuals=y)           #kurtosis
# 			spm1.kurtosis = k
# 	else:
# 		Q         = y.shape[1]
# 		s,k,z0,z1 = np.array( [mr_single_node(y[:,i,:], skewness, kurtosis)   for i in range(Q)] ).T
# 		fwhm      = estimate_mvfwhm(y)
# 		resels    = mv_resel_counts(y, fwhm, roi=None)
# 		### build SPM object(s):
# 		spm0,spm1 = None,None
# 		if skewness:
# 			spm0  = SPM_X2(z0, (1,v), fwhm, resels, residuals=y)   #skewness
# 		if kurtosis:
# 			spm1  = SPM_Z(z1, fwhm, resels, residuals=y)         #kurtosis
#
# 	return _assemble_results(spm0, spm1)
#



# def hotellings(y, skewness=True, kurtosis=False):
# 	r   = y - y.mean(axis=0)
# 	return residuals(r, skewness, kurtosis)









