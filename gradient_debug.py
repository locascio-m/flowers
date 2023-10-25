import flowers_interface as flow
import numpy as np
import tools as tl
import visualization as viz
import matplotlib.pyplot as plt
import old_files.model as set
from scipy.interpolate import NearestNDInterpolator
import floris.tools as wfct

import warnings
warnings.filterwarnings("ignore")

wr = tl.load_wind_rose(6) # 14
# wr["wd"] = np.remainder(450 - wr.wd, 360)
# wr.sort_values("wd", inplace=True)

# layout_x, layout_y = tl.discrete_layout(6, idx=12)

# layout_x = 126. * np.array([0.,7.,14.])
# layout_y = 126. * np.array([0.,0.,0.])
# layout_x = 126. * np.array([0.,0.,7.,7.])
# layout_y = 126. * np.array([0.,7.,0.,7.])
layout_x = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])


fi = flow.FlowersInterface(wr,layout_x,layout_y, num_terms=10)
aep, grad = fi.calculate_aep(gradient=True)

fd = np.zeros((len(layout_x),2))
h = 126.*1e-8

for i in range(len(layout_x)):
    layout_xx = np.copy(layout_x)
    layout_xx[i] += h
    fi.reinitialize(layout_x=layout_xx)
    fd1 = fi.calculate_aep()
    layout_xx[i] -= 2*h
    fi.reinitialize(layout_x=layout_xx)
    fd2 = fi.calculate_aep()

    fd[i,0] = (fd1 - fd2)/(2*h)


for i in range(len(layout_x)):
    layout_yy = np.copy(layout_y)
    layout_yy[i] += h
    fi.reinitialize(layout_y=layout_yy)
    fd1 = fi.calculate_aep()
    layout_yy[i] -= 2*h
    fi.reinitialize(layout_y=layout_yy)
    fd2 = fi.calculate_aep()

    fd[i,1] = (fd1 - fd2)/(2*h)

print(grad)
# print(fd)
print(grad/aep*1e5)
print(np.abs((grad-fd)/fd)*100)
print(np.sign(grad/fd))

viz.plot_wind_rose(wr)
viz.plot_layout(layout_x,layout_y)
plt.show()