
import numpy as np
from matplotlib import pyplot
import spm1d



#(0) Load dataset:
dataset = spm1d.data.uv0d.ci2.AnimalsInResearch()
yA,yB   = dataset.get_data()
print( dataset )



#(1) Compute confidence intervals:
np.random.seed(0)
alpha      = 0.05
iterations = 1000
ci         = spm1d.stats.ci_twosample(yA, yB, alpha, datum='meanA', mu='meanB')
cinp       = spm1d.stats.nonparam.ci_twosample(yA, yB, alpha, datum='meanA', mu='meanB', iterations=iterations)
print( ci )
print( cinp )



#(2) Plot the CIs:
pyplot.close('all')
pyplot.figure(figsize=(8,4))
pyplot.get_current_fig_manager().window.move(0, 0)
ax0 = pyplot.subplot(121);  ci.plot(ax0);    ax0.set_title('Parametric', size=10)
ax1 = pyplot.subplot(122);  cinp.plot(ax1);  ax1.set_title('Non-parametric', size=10)
pyplot.suptitle('Two-sample CIs')
pyplot.show()