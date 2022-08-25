# FLOWERS

# Michael LoCascio

import numpy as np
import flowers_interface as flow
import tools as tl
import matplotlib.pyplot as plt
import visualization as vis
from shapely.geometry import Polygon, Point, LineString

"""
This file is a workspace for testing the vectorization of the FLOWERS
code and the implementation of automatic differentiation.
"""


boundaries = [
    (2714.4, 4049.4),
    (2132.7, 938.8),
    (2092.8, 591.6),
    (2078.9, 317.3),
    (2076.1, 148.5),
    (2076.6, 0.0),
    (2076.5, 6.5),
    (1208.6, 847.0),
    (0.0, 2017.7),
    (1496.7, 4027.2),
    (1531.8, 4006.2),
    (1931.2, 3818.5),
    (2058.3, 3783.6),
    (2192.8, 3792.9),
    (2316.8, 3846.4),
    (2416.0, 3939.1),
    (2528.6, 4089.0),
    (2550.9, 4126.3)
]
boundary_polygon = Polygon(boundaries)
boundary_line = LineString(boundaries)
xmin = np.min([tup[0] for tup in boundaries])
xmax = np.max([tup[0] for tup in boundaries])
ymin = np.min([tup[1] for tup in boundaries])
ymax = np.max([tup[1] for tup in boundaries])

x = np.linspace(xmin, xmax, 1000)
y = np.linspace(ymin, ymax, 1000)
X, Y = np.meshgrid(x,y)
xx = X.flatten()
yy = Y.flatten()
d = np.zeros(len(xx))
for i in range(len(xx)):
    loc = Point(xx[i], yy[i])
    d[i] = loc.distance(boundary_line) #NaNsafe, or 1 to 5 m inside boundary
D = np.reshape(d, np.shape(X))

fig, ax = plt.subplots(1,1)
c = ax.pcolormesh(X, Y, D)
ax.set(xlabel="x [m]", ylabel="y [m]", title='Distance', aspect='equal')
fig.colorbar(c, ax=ax)

# Number of turbines
# n_turb = 31

# # Wind rose resolution
# num_terms = 181

# # Randomize wind farm layout
# layout_x = 126.*np.array([0., 5., 10.])
# layout_y = np.array([0., 0., 0.])
# xx = np.linspace(-5*126., 16*126., 400)
# yy = np.linspace(-2*126., 2*126., 100)
# X, Y = np.meshgrid(xx,yy)

# # Initialize optimization interface

# fi = flow.Flowers(wind_rose,layout_x,layout_y,k=0.05,D=126.0)
# fi.fourier_coefficients(num_terms=num_terms)
# u = fi.calculate_field(X,Y)

# fig, ax = plt.subplots(1,1)

# im = ax.pcolormesh(X, Y, u, cmap='coolwarm')
# ax.set(aspect='equal')
# vis.plot_wind_rose(wind_rose)
# ax.plot(fi.layout_x, fi.layout_y, 'ow')
plt.show()