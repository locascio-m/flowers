import numpy as np
import pickle
import tools as tl
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import model as set

# # Load layout, boundaries, and rose
# layout_x, layout_y = tl.load_layout('iea')

# xmin = -126.
# xmax = np.max(layout_x)
# ymin = -126.
# ymax = np.max(layout_y)

# boundaries = [
#     (2714.4, 4049.4),
#     (2132.7, 938.8),
#     (2092.8, 591.6),
#     (2078.9, 317.3),
#     (2076.1, 148.5),
#     (2076.6, 0.0),
#     (2076.5, 6.5),
#     (1208.6, 847.0),
#     (0.0, 2017.7),
#     (1496.7, 4027.2),
#     (1531.8, 4006.2),
#     (1931.2, 3818.5),
#     (2058.3, 3783.6),
#     (2192.8, 3792.9),
#     (2316.8, 3846.4),
#     (2416.0, 3939.1),
#     (2528.6, 4089.0),
#     (2550.9, 4126.3)
# ]

# wind_rose = tl.load_wind_rose(6)

# # Initialize computations
# geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
# geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1)
# freq_floris = geo.freq_floris
                
# fi = geo.flowers
# pi = geo.floris

# geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
# geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1)
# gi = geo.floris

# nx = 90
# ny = 135

# xx = np.linspace(xmin,xmax,nx,endpoint=True)
# yy = np.linspace(ymin,ymax,ny,endpoint=True)
# X,Y = np.meshgrid(xx,yy)

# flowers_aep = np.zeros_like(X)
# park_aep = np.zeros_like(X)
# gauss_aep = np.zeros_like(X)
# infeasible = np.zeros_like(X)

# poly = Polygon(boundaries)

# # Move WT12 around domain and compute AEP
# for i in range(ny):
#     print(i)
#     for j in range(nx):
#         layout_x[12] = X[i,j]
#         layout_y[12] = Y[i,j]

#         fi.layout_x = layout_x
#         fi.layout_y = layout_y

#         pi.reinitialize(layout=(layout_x.flatten(),layout_y.flatten()),time_series=True)
#         gi.reinitialize(layout=(layout_x.flatten(),layout_y.flatten()),time_series=True)

#         pt = Point(X[i,j], Y[i,j])
#         if np.any(np.sqrt((layout_x[12] - np.delete(layout_x,12))**2 + (layout_y[12] - np.delete(layout_y,12))**2) < 63.) or not pt.within(poly):
#             infeasible[i,j] = 1

#         flowers_aep[i,j] = fi.calculate_aep()

#         pi.calculate_wake()
#         park_aep[i,j] = np.sum(pi.get_farm_power() * freq_floris * 8760)

#         gi.calculate_wake()
#         gauss_aep[i,j] = np.sum(gi.get_farm_power() * freq_floris * 8760)

# flowers_aep = np.ma.masked_where(infeasible,flowers_aep)
# park_aep = np.ma.masked_where(infeasible,park_aep)
# gauss_aep = np.ma.masked_where(infeasible,gauss_aep)

# flowers_aep /= np.amax(flowers_aep)
# park_aep /= np.amax(park_aep)
# gauss_aep /= np.amax(gauss_aep)

# pickle.dump((X, Y, flowers_aep, park_aep, gauss_aep), open('solutions/features_smooth.p','wb'))

# Mutation
# Load layout
xx, yy = tl.load_layout('iea')
xx *= 1.25
yy *= 1.25

wr = tl.load_wind_rose(6)
nt = 31

# Define mutation parameters
N = 50
step = 126./(4*np.sqrt(2))
np.random.seed(3)

# Store AEP
flowers_aep = np.zeros(N+1)
park_aep= np.zeros(N+1)
gauss_aep = np.zeros(N+1)

geo = set.ModelComparison(wr, xx, yy, model='park')
aep, _ = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1, display=False)
flowers_norm = aep[0]
park_norm = aep[1]

freq_floris = geo.freq_floris     
fi = geo.flowers
pi = geo.floris

geo = set.ModelComparison(wr, xx, yy, model='gauss')
aep, _ = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1, display=False)
gauss_norm = aep[1]

gi = geo.floris

flowers_aep[0] = 1.
park_aep[0] = 1.
gauss_aep[0] = 1.

x_all = [xx]
y_all = [yy]

for i in np.arange(1,N+1):
    print(i)
    x0 = xx + np.random.normal(0.,step,nt)
    y0 = yy + np.random.normal(0.,step,nt)

    fi.layout_x = x0
    fi.layout_y = y0

    pi.reinitialize(layout=(x0.flatten(),y0.flatten()),time_series=True)
    gi.reinitialize(layout=(x0.flatten(),y0.flatten()),time_series=True)

    flowers_aep[i] = fi.calculate_aep() / flowers_norm

    pi.calculate_wake()
    park_aep[i] = np.sum(pi.get_farm_power() * freq_floris * 8760) / park_norm

    gi.calculate_wake()
    gauss_aep[i] = np.sum(gi.get_farm_power() * freq_floris * 8760) / gauss_norm

    xx = x0
    yy = y0

    x_all.append(x0)
    y_all.append(y0)

fig, (ax0,ax1) = plt.subplots(1,2, figsize=(11,7))
ax0.set(aspect='equal', xlim=[-5,35], ylim=[-5,45], xlabel='x/D', ylabel='y/D')
ax1.set(xlim=[0,N], ylim=[0.96,1.04], xlabel='Iteration', ylabel='Normalized AEP')

line0, = ax0.plot([],[],"o",color='tab:red',markersize=7)
line1, = ax1.plot([],[],"-o",markersize=3)
line2, = ax1.plot([],[],"-o",markersize=3)
line3, = ax1.plot([],[],"-o",markersize=3)

# Function to update turbine positions
def animate(i):
    line0.set_data(x_all[i]/126., y_all[i]/126.)
    line1.set_data(range(i+1),flowers_aep[0:i+1])
    line2.set_data(range(i+1),park_aep[0:i+1])
    line3.set_data(range(i+1),gauss_aep[0:i+1])
    ax1.legend(['FLOWERS','Conventional-Park','Conventional-Gauss'])
    return line0, line1, line2, line3

# Animation
ani = animation.FuncAnimation(fig, animate, frames=N+1, repeat=False)

ani.save('mutation.mp4')
# plt.show()