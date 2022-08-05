# FLOWERS

# Michael LoCascio

import numpy as np
import flowers_interface as flow
import tools as tl
import matplotlib.pyplot as plt
import visualization as vis

# Wind rose (sampled from stored wind roses)
wind_rose = tl.load_wind_rose(2)

# Number of turbines
n_turb = 31

# Wind rose resolution
num_terms = 181

# Randomize wind farm layout
layout_x = 126.*np.array([0., 5., 10.])
layout_y = np.array([0., 0., 0.])
xx = np.linspace(-5*126., 16*126., 400)
yy = np.linspace(-2*126., 2*126., 100)
X, Y = np.meshgrid(xx,yy)

# Initialize optimization interface

fi = flow.Flowers(wind_rose,layout_x,layout_y,k=0.05,D=126.0)
fi.fourier_coefficients(num_terms=num_terms)
u = fi.calculate_field(X,Y)

fig, ax = plt.subplots(1,1)

im = ax.pcolormesh(X, Y, u, cmap='coolwarm')
ax.set(aspect='equal')
vis.plot_wind_rose(wind_rose)
# ax.plot(fi.layout_x, fi.layout_y, 'ow')
plt.show()