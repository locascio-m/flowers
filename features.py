import numpy as np
import pickle
import tools as tl
from shapely.geometry import Polygon, Point

import model as set

# Load layout, boundaries, and rose
layout_x, layout_y = tl.load_layout('iea')

xmin = np.min(layout_x)
xmax = np.max(layout_x)
ymin = np.min(layout_y)
ymax = np.max(layout_y)

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

wind_rose = tl.load_wind_rose(1)

# Initialize computations
nx = 90
ny = 135
xx = np.linspace(xmin,xmax,nx)
yy = np.linspace(ymin,ymax,ny)
X,Y = np.meshgrid(xx,yy)

flowers_aep = np.zeros_like(X)
floris_aep = np.zeros_like(X)

poly = Polygon(boundaries)

# Move WT12 around domain and compute AEP
for i in range(nx):
    for j in range(ny):
        layout_x[12] = X[i,j]
        layout_y[12] = Y[i,j]
        pt = Point(X[i,j], Y[i,j])
        if np.any(np.sqrt((layout_x[12] - np.delete(layout_x,12))**2 + (layout_y[12] - np.delete(layout_y,12))**2) < 63.) or not pt.within(poly):
            aep = (0.,0.)
        else:
            geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
            aep, tmp = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1, display=False)
        flowers_aep[i,j] = aep[0]
        floris_aep[i,j] = aep[1]

flowers_aep /= np.max(flowers_aep)
floris_aep /= np.max(floris_aep)

pickle.dump((X, Y, flowers_aep, floris_aep), open('solutions/features_aep.p','wb'))
