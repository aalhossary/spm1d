
import os
import numpy as np
from .. import _base



_note   = '\n'
_note   = '--------LICENSE NOTE--------\n'
_note  += 'No license was provided with the MVNweb source code and datasets when downloaded from:\n\n'
_note  += 'https://github.com/selcukorkmaz/MVNweb\n\n'
_note  += 'on 2017.02.25. Data are redistributed here assuming that the authors have provided consent to do so. '
_note  += 'If you use these data please cite the original paper (Korkmaz et al. 2014): which is available here:\n\n'
_note  += 'https://journal.r-project.org/archive/2014-2/korkmaz-goksuluk-zararsiz.pdf\n'




class _DatasetNormMV(_base.DatasetNormalityMV):
	def __init__(self, testname='mardia'):
		super(_DatasetNormMV, self).__init__()
		self.testname  = testname
		self._set_expected_results(self.testname)
		self.df       = None



class Iris(_DatasetNormMV):
	def _set_values(self):
		self.datafile = os.path.join(_base.get_datafilepath(), 'mvnorm_Iris.npy')
		self.www      = 'https://github.com/selcukorkmaz/MVNweb'
		self.Y        = np.load(self.datafile)
		_note1        = 'This is the "iris2.txt" dataset from the MVNweb tool available at:\n\n'
		_note1       += 'https://github.com/selcukorkmaz/MVNweb\n\n'
		_note1       += 'The data are originally from Johnson & Wichern (1992, p. 562, Table 11.5: "Iris data"). '
		_note1       += 'Following both MVNweb and Roystest.m (Trujillo-Ortiz et al., 2007) only the setosa data are included here.\n\n'
		_note1       += 'REFERENCES:\n'
		_note1       += '[1]  Johnson, R.A. and Wichern, D. W. (1992). Applied Multivariate Statistical Analysis. 3rd. ed. New-Jersey:Prentice Hall.\n\n'
		_note1       += '[2]  Korkmaz S, Goksuluk D, Zararsiz G (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 2014 6(2):151-162.\n\n'
		_note1       += "[3]  Trujillo-Ortiz A, Hernandez-Walls A, Barba-Rojo K, Cupul-Magana L (2007). Roystest:Royston's Multivariate Normality Test.  A MATLAB file. [WWW document]. URL http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=17811\n\n"
		self.note     = 'Note     ', _note1 + _note
		self.cite     = '[Korkmaz S, Goksuluk D, Zararsiz G (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 2014 6(2):151-162.'

	def _set_expected_results(self, testname):
		if testname == 'henze':
			self.z    = 0.9488453
			self.p    = 0.04995356
		elif testname == 'mardia':
			self.z    = 27.85973
			self.p    = 0.1127617
		elif testname == 'royston':
			self.z    = 31.51803
			self.p    = 2.187653e-6



class IrisTwoComponents(Iris):
	def _set_values(self):
		super(IrisTwoComponents, self)._set_values()
		self.Y        = self.Y[:,:2]
		_note1        = '(Identical to the "Iris" dataset described below, but only the first two (of four) vector components used.)\n\n'
		self.note     = 'Note     ', _note1 + self.note[1]

	def _set_expected_results(self, testname):
		if testname == 'henze':
			self.z    = 0.2856007
			self.p    = 0.9146336
		elif testname == 'mardia':
			self.z    = 0.8379339
			self.p    = 0.9332914
		elif testname == 'royston':
			self.z    = 2.698277
			self.p    = 0.2445737



class MVNWebBivariate(_DatasetNormMV):
	def _set_values(self):
		self.datafile = os.path.join(_base.get_datafilepath(), 'mvnorm_MVNWebBivariate.npy')
		self.www      = 'https://github.com/selcukorkmaz/MVNweb'
		self.Y        = np.load(self.datafile)
		self.cite     = 'Korkmaz S, Goksuluk D, Zararsiz G (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 2014 6(2):151-162'
		_note1        = 'This is the "bivariate.txt" dataset from the MVNweb tool available at:\n\n'
		_note1       += 'https://github.com/selcukorkmaz/MVNweb\n\n'
		_note1       += 'REFERENCES:\n'
		_note1       += '%s.\n\n' %self.cite
		self.note     = 'Note     ', _note1 + _note

	def _set_expected_results(self, testname):
		if testname == 'henze':
			self.z    = 0.6564342
			self.p    = 0.6723063
		elif testname == 'mardia':
			self.z    = 0.9243787
			self.p    = 0.9210375
		elif testname == 'royston':
			self.z    = 0.3986913
			self.p    = 0.8192667







class MVNWebMultivariate(_DatasetNormMV):
	def _set_values(self):
		self.datafile = os.path.join(_base.get_datafilepath(), 'mvnorm_MVNWebMultivariate.npy')
		self.www      = 'https://github.com/selcukorkmaz/MVNweb'
		self.Y        = np.load(self.datafile)
		self.cite     = 'Korkmaz S, Goksuluk D, Zararsiz G (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 2014 6(2):151-162'
		_note1        = 'This is the "multivariate.txt" dataset from the MVNweb tool available at:\n\n'
		_note1       += 'https://github.com/selcukorkmaz/MVNweb\n\n'
		_note1       += 'REFERENCES:\n'
		_note1       += '%s.\n\n' %self.cite
		self.note     = 'Note     ', _note1 + _note

	def _set_expected_results(self, testname):
		if testname == 'henze':
			self.z    = 0.7807283
			self.p    = 0.747874
		elif testname == 'mardia':
			self.z    = 8.948027
			self.p    = 0.537043
		elif testname == 'royston':
			self.z    = 1.552225
			self.p    = 0.6702702





# class RSXLMultivariate(_base.DatasetNormalityMV):
# 	def _set_values(self):
# 		self.www  = 'http://www.real-statistics.com/multivariate-statistics/multivariate-normal-distribution/multivariate-normality-testing/'
# 		y0        = [36,30,28,21,16,   15,20,21,19,17,   16,32,27,24,28,   40,41,39,23]
# 		y1        = [33,27,29,15,16,   18,20,22,20,15,   15,23,23,26,34,   33,40,28,19]
# 		self.Y    = np.array([y0,y1]).T
# 		self.z    = 0.66
# 		self.df   = 1,2
# 		self.p    = 0.7195
# 		self.note = 'Note     ', 'Original data are from p.194 of:  Kendall M (1948) Rank Correlation Methods, Charles Griffin & Company Ltd.'

