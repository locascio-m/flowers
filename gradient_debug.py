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


fi = flow.FlowersInterface(wr,layout_x,layout_y)
aep, grad = fi.calculate_aep(gradient=True)

# grad = fi.calculate_aep_gradient()


# fli = wfct.floris_interface.FlorisInterface('./input/park.yaml')
# fli.reinitialize(
#     layout_x=layout_x.flatten(),
#     layout_y=layout_y.flatten(), 
#     wind_shear=0)
# # Initialize wind direction-speed frequency array for AEP
# wd_array = np.array(wr["wd"].unique(), dtype=float)
# ws_array = np.array(wr["ws"].unique(), dtype=float)
# wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
# freq_interp = NearestNDInterpolator(wr[["wd", "ws"]], wr["freq_val"])
# freq = freq_interp(wd_grid, ws_grid)
# freq_floris = freq / np.sum(freq)

# fli.reinitialize(
#     wind_directions=wd_array,
#     wind_speeds=ws_array)
# fli.calculate_wake()
# aep_floris = np.sum(fli.get_farm_power() * freq_floris * 8760)
# print(aep_floris)
# dasd


# fi.reinitialize(num_terms=10)
# aep, grad_new = fi.calculate_aep(gradient=True)
# print(grad)
# print(grad_new)
# print((grad-grad_new)/grad*100)
# dasd

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
    # fd[i,0] = (fd1 - aep)/h


for i in range(len(layout_x)):
    layout_yy = np.copy(layout_y)
    layout_yy[i] += h
    fi.reinitialize(layout_y=layout_yy)
    fd1 = fi.calculate_aep()
    layout_yy[i] -= 2*h
    fi.reinitialize(layout_y=layout_yy)
    fd2 = fi.calculate_aep()

    fd[i,1] = (fd1 - fd2)/(2*h)
    # fd[i,1] = (fd1 - aep)/h

# fd_new = np.zeros((len(layout_x),2))
# fi.fourier_coefficients(num_terms=5)
# aep_new = fi.calculate_aep()

# for i in range(len(layout_x)):
#     layout_xx = np.copy(layout_x)
#     layout_xx[i] += h
#     tmp = flow.Flowers(wr,layout_xx,layout_y)
#     tmp.fourier_coefficients(num_terms=5)
#     fd1 = tmp.calculate_aep()

#     layout_xx[i] -= 2*h
#     tmp = flow.Flowers(wr,layout_xx,layout_y)
#     tmp.fourier_coefficients(num_terms=5)
#     fd2 = tmp.calculate_aep()

#     fd_new[i,0] = (fd1 - fd2)/(2*h)

# for i in range(len(layout_x)):
#     layout_yy = np.copy(layout_y)
#     layout_yy[i] += h
#     tmp = flow.Flowers(wr,layout_x,layout_yy)
#     tmp.fourier_coefficients(num_terms=5)
#     fd1 = tmp.calculate_aep()

#     layout_yy[i] -= 2*h
#     tmp = flow.Flowers(wr,layout_x,layout_yy)
#     tmp.fourier_coefficients(num_terms=5)
#     fd2 = tmp.calculate_aep()

#     fd_new[i,1] = (fd1 - fd2)/(2*h)

# print(fd/aep*100)
# print(fd_new/aep_new*100)
# print((fd_new-fd)/fd*100)

print(grad)
print(fd)
# print(grad/aep*100)
print(np.abs((grad-fd)/fd)*100)
print(np.sign(grad/fd))

viz.plot_wind_rose(wr)
viz.plot_layout(layout_x,layout_y)
plt.show()