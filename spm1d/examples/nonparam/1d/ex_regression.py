
import numpy as np
from matplotlib import pyplot
import spm1d




#(0) Load dataset:
dataset    = spm1d.data.uv1d.regress.SimulatedPataky2015c()
dataset    = spm1d.data.uv1d.regress.SpeedGRF()
y,x        = dataset.get_data()



#(1) Conduct non-parametric test:
np.random.seed(0)
alpha      = 0.05
two_tailed = False
snpm       = spm1d.stats.nonparam.regress(y, x)
snpmi      = snpm.inference(alpha, two_tailed=two_tailed, iterations=500)
print snpmi
print snpmi.clusters



#(2) Compare with parametric result:
spm        = spm1d.stats.regress(y, x)
spmi       = spm.inference(alpha, two_tailed=two_tailed)
print spmi
print spmi.clusters



#(3) Plot
pyplot.close('all')
pyplot.figure(figsize=(12,4))
pyplot.get_current_fig_manager().window.move(0, 0)
ax0 = pyplot.subplot(121)
ax1 = pyplot.subplot(122)
for ax,zi in zip([ax0,ax1], [spmi,snpmi]):
	zi.plot(ax=ax)
	zi.plot_threshold_label(ax=ax, fontsize=8)
	zi.plot_p_values(ax=ax, size=10)
pyplot.show()


