
'''
Multivariate normality testing

All procedures fit the given model, calculate residuals then use
"normality.residuals" to conduct a Mardia-Rencher test
on the multivariate residuals.
'''

# Copyright (C) 2016  Todd Pataky

from math import log
import numpy as np
from .. _spm import SPM0D_X2, SPM_X2
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



def mr_single_node(y):
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
	J,I     = y.shape
	J,I     = float(J), float(I)
	### compute SIGMA:
	m       = y.mean(axis=0)
	D       = np.matrix(y - m)
	SIGMA   = np.cov( D.T, ddof=0)
	SIGMAi  = np.matrix( np.linalg.inv( SIGMA ) )
	### compute skewness:
	zsum    = np.array( D * SIGMAi * D.T )
	### calculation from Mardia (1970) is:
	### z       = (z**3).sum() / (6.0 * J)
	### better calculation (especially for small samples)
	### from Rencher & Christensen (2012, p.108) is:
	z       = (zsum**3).sum() / (J**2)
	z0      = (I+1) * (J+1) * (J+3)
	z1      = 6 * ( (I+1) * (J+1) - 6 )
	a       = (z0 / z1) * z
	### compute kurtosis:
	b       = (np.diag(zsum)**2).sum() / J
	if J >= 500:   #Rencher & Christensen (2012, p.108, eqn. 4.39)
		b  -= I * (I + 2)
		b  *= (   J / ( 8 * I * (I+2) )   )**0.5
	else:   #Rencher & Christensen (2012, p.108, Eqn. 4.40)
		### NOTE!!!   Eqn.4.40 uses (J + I + 1) but (J - I - 1) gives better results
		b  -= I * (I + 2) * (J - I - 1) / J
		b  *= (   (J-1) / ( 8 * I * (I+2)  )   )**0.5
	return a,b
	

#
# def residuals(y):
# 	'''
# 	Compute the Mardia-Rencher test statistics for skewness and kurtosis test statistics
# 	for a set of model residuals.
# 	'''
# 	J  = y.shape[0]
# 	if J < 8:
# 		raise( ValueError('In order to conduct a normality test there must at least 8 observations. Only %d found.' %J)   )
# 	df     = 1, 2
# 	if np.ndim(y)==1:
# 		a,b    = mr_single_node(y)
# 		spm    = SPM0D_X2(a, df, residuals=y)
# 	else:
# 		k2     = np.array( [k2_single_node(yy)   for yy in y.T] )
# 		fwhm   = rft1d.geom.estimate_fwhm(y)
# 		resels = rft1d.geom.resel_counts(y, fwhm, element_based=False)
# 		spm    = SPM_X2(k2, df, fwhm, resels, residuals=y)
# 	return spm




# def _stack_data(*args):
# 	return np.hstack(args) if (np.ndim(args[0])==1) else np.vstack(args)
#
#
# def anova1(y, A):
# 	spm  = _main_anova1(y, A)
# 	return residuals( spm.residuals )
# def anova1rm(y, A, SUBJ):
# 	spm  = _main_anova1rm(y, A, SUBJ, _force_approx0D=True)
# 	return residuals( spm.residuals )
#
# def anova2(y, A, B):
# 	spm  = _main_anova2(y, A, B)
# 	return residuals( spm[0].residuals )
# def anova2nested(y, A, B):
# 	spm  = _main_anova2nested(y, A, B)
# 	return residuals( spm[0].residuals )
# def anova2onerm(y, A, B, SUBJ):
# 	spm  = _main_anova2onerm(y, A, B, SUBJ, _force_approx0D=True)
# 	return residuals( spm[0].residuals )
# def anova2rm(y, A, B, SUBJ):
# 	spm  = _main_anova2rm(y, A, B, SUBJ, _force_approx0D=True)
# 	return residuals( spm[0].residuals )
#
#
# def anova3(y, A, B, C):
# 	spm  = _main_anova3(y, A, B, C)
# 	return residuals( spm[0].residuals )
# def anova3nested(y, A, B, C):
# 	spm  = _main_anova3nested(y, A, B, C)
# 	return residuals( spm[0].residuals )
# def anova3onerm(y, A, B, C, SUBJ):
# 	spm  = _main_anova3onerm(y, A, B, C, SUBJ, _force_approx0D=True)
# 	return residuals( spm[0].residuals )
# def anova3tworm(y, A, B, C, SUBJ):
# 	spm  = _main_anova3tworm(y, A, B, C, SUBJ, _force_approx0D=True)
# 	return residuals( spm[0].residuals )
# def anova3rm(y, A, B, C, SUBJ):
# 	spm  = _main_anova3rm(y, A, B, C, SUBJ, _force_approx0D=True)
# 	return residuals( spm[0].residuals )
#
#
#
# def regress(y, x):
# 	spm  = _main_regress(y, x)
# 	return residuals( spm.residuals )
#
#
# def ttest(y):
# 	r   = y - y.mean(axis=0)
# 	return residuals(r)
#
# def ttest_paired(yA, yB):
# 	return ttest( yA - yB )
#
# def ttest2(yA, yB):
# 	rA   = yA - yA.mean(axis=0)
# 	rB   = yB - yB.mean(axis=0)
# 	r    = _stack_data(rA, rB)
# 	return residuals(r)







