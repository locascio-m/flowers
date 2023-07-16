import numpy as np
import pickle
import tools as tl
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import model as set


# Define inputs 1,2,3,4,5,6,8,10,12,15,18
model = 'gauss'
terms = 18

# Load layout, boundaries, and rose
layout_x, layout_y = tl.load_layout('iea')

xmin = -126.
xmax = np.max(layout_x) + 126.
ymin = -126.
ymax = np.max(layout_y) + 126.

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

wind_rose = tl.load_wind_rose(6)

file_name = 'solutions/smooth_' + model + str(terms) + '.p'

# Initialize computations
if model == 'flowers':
    geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
    geo.compare_aep(num_terms=terms, wd_resolution=5.0, ws_avg=True, iter=1)
    fi = geo.flowers
else:
    geo = set.ModelComparison(wind_rose, layout_x, layout_y, model=model)
    geo.compare_aep(num_terms=5, wd_resolution=terms, ws_avg=True, iter=1)
    freq_floris = geo.freq_floris
    fi = geo.floris

nx = 118
ny = 170

xx = np.linspace(xmin,xmax,nx,endpoint=True)
yy = np.linspace(ymin,ymax,ny,endpoint=True)
X,Y = np.meshgrid(xx,yy)

aep = np.zeros_like(X)
infeasible = np.zeros_like(X)

poly = Polygon(boundaries)

# Move WT12 around domain and compute AEP
for i in range(ny):
    print(i)
    for j in range(nx):
        layout_x[12] = X[i,j]
        layout_y[12] = Y[i,j]

        if model == 'flowers':
            fi.layout_x = layout_x
            fi.layout_y = layout_y
            aep[i,j] = fi.calculate_aep()
        
        else:
            fi.reinitialize(layout_x=layout_x.flatten(),layout_y=layout_y.flatten(),time_series=True)
            fi.calculate_wake()
            aep[i,j] = np.sum(fi.get_farm_power() * freq_floris * 8760)

        pt = Point(X[i,j], Y[i,j])
        if np.any(np.sqrt((layout_x[12] - np.delete(layout_x,12))**2 + (layout_y[12] - np.delete(layout_y,12))**2) < 63.) or not pt.within(poly):
            infeasible[i,j] = 1


aep = np.ma.masked_where(infeasible,aep)

aep /= np.amax(aep)

pickle.dump((X, Y, aep), open(file_name,'wb'))

# # Mutation
# # Load layout
# xx, yy = tl.load_layout('iea')
# xx *= 1.25
# yy *= 1.25

# wr = tl.load_wind_rose(6)
# nt = 31

# # Define mutation parameters
# N = 25
# step = 126./(5)
# np.random.seed(4)

# # Store AEP
# flowers_100 = np.ones(N+1)
# flowers_25 = np.ones(N+1)
# flowers_5 = np.ones(N+1)
# park_10 = np.ones(N+1)
# park_5 = np.ones(N+1)
# park_1 = np.ones(N+1)
# gauss_10 = np.ones(N+1)
# gauss_5 = np.ones(N+1)
# gauss_1 = np.ones(N+1)

# geo = set.ModelComparison(wr, xx, yy, model='park')
# aep, _ = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1, display=False)
# flowers_5_norm = aep[0]
# park_5_norm = aep[1]

# freq_5 = geo.freq_floris     
# fi_5 = geo.flowers
# pi_5 = geo.floris

# geo = set.ModelComparison(wr, xx, yy, model='park')
# aep, _ = geo.compare_aep(num_terms=25, wd_resolution=1.0, ws_avg=True, iter=1, display=False)
# flowers_25_norm = aep[0]
# park_1_norm = aep[1]
# freq_1 = geo.freq_floris
# fi_25 = geo.flowers
# pi_1 = geo.floris

# geo = set.ModelComparison(wr, xx, yy, model='park')
# aep, _ = geo.compare_aep(num_terms=100, wd_resolution=10.0, ws_avg=True, iter=1, display=False)
# flowers_100_norm = aep[0]
# park_10_norm = aep[1]
# freq_10 = geo.freq_floris   
# fi_100 = geo.flowers  
# pi_10 = geo.floris

# geo = set.ModelComparison(wr, xx, yy, model='gauss')
# aep, _ = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, iter=1, display=False)
# gauss_5_norm = aep[1]
# gi_5 = geo.floris

# geo = set.ModelComparison(wr, xx, yy, model='gauss')
# aep, _ = geo.compare_aep(num_terms=5, wd_resolution=1.0, ws_avg=True, iter=1, display=False)
# gauss_1_norm = aep[1]    
# gi_1 = geo.floris

# geo = set.ModelComparison(wr, xx, yy, model='gauss')
# aep, _ = geo.compare_aep(num_terms=5, wd_resolution=10.0, ws_avg=True, iter=1, display=False)
# gauss_10_norm = aep[1]    
# gi_10 = geo.floris

# x_all = [xx]
# y_all = [yy]

# for i in np.arange(1,N+1):
#     print(i)
#     x0 = xx + np.random.normal(0.,step,nt)
#     y0 = yy + np.random.normal(0.,step,nt)

#     fi_100.layout_x = x0
#     fi_100.layout_y = y0
#     fi_25.layout_x = x0
#     fi_25.layout_y = y0
#     fi_5.layout_x = x0
#     fi_5.layout_y = y0

#     pi_1.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)
#     pi_5.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)
#     pi_10.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)
#     gi_1.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)
#     gi_5.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)
#     gi_10.reinitialize(layout_x=x0.flatten(),layout_y=y0.flatten(),time_series=True)

#     flowers_5[i] = fi_5.calculate_aep() / flowers_5_norm
#     flowers_25[i] = fi_25.calculate_aep() / flowers_25_norm
#     flowers_100[i] = fi_100.calculate_aep() / flowers_100_norm

#     pi_1.calculate_wake()
#     park_1[i] = np.sum(pi_1.get_farm_power() * freq_1 * 8760) / park_1_norm

#     pi_5.calculate_wake()
#     park_5[i] = np.sum(pi_5.get_farm_power() * freq_5 * 8760) / park_5_norm

#     pi_10.calculate_wake()
#     park_10[i] = np.sum(pi_10.get_farm_power() * freq_10 * 8760) / park_10_norm

#     gi_1.calculate_wake()
#     gauss_1[i] = np.sum(gi_1.get_farm_power() * freq_1 * 8760) / gauss_1_norm

#     gi_5.calculate_wake()
#     gauss_5[i] = np.sum(gi_5.get_farm_power() * freq_5 * 8760) / gauss_5_norm

#     gi_10.calculate_wake()
#     gauss_10[i] = np.sum(gi_10.get_farm_power() * freq_10 * 8760) / gauss_10_norm

#     xx = x0
#     yy = y0

#     x_all.append(x0)
#     y_all.append(y0)

# fig = plt.figure(figsize=(12,7))
# ax0 = plt.subplot2grid((3,2),(0,0),rowspan=3)
# ax1 = plt.subplot2grid((3,2),(0,1))
# ax2 = plt.subplot2grid((3,2),(1,1))
# ax3 = plt.subplot2grid((3,2),(2,1))

# # fig, (ax0,ax1) = plt.subplots(1,2, )
# ax0.set(aspect='equal', xlim=[-5,35], ylim=[-5,45], xlabel='x/D', ylabel='y/D')
# ax1.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
# ax2.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
# ax3.set(xlim=[0,N], ylim=[0.975,1.025], xlabel='Iteration', ylabel='Normalized AEP')

# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# line0, = ax0.plot([],[],"o",color='tab:red',markersize=7)
# line1, = ax1.plot([],[],"-o",color=colors[0],markersize=3)
# line2, = ax1.plot([],[],":o",color=colors[1],markersize=3)
# line3, = ax1.plot([],[],"--o",color=colors[2],markersize=3)
# line4, = ax2.plot([],[],"-o",color=colors[4],markersize=3)
# line5, = ax2.plot([],[],":o",color=colors[5],markersize=3)
# line6, = ax2.plot([],[],"--o",color=colors[6],markersize=3)
# line7, = ax3.plot([],[],"-o",color=colors[8],markersize=3)
# line8, = ax3.plot([],[],":o",color=colors[9],markersize=3)
# line9, = ax3.plot([],[],"--o",color=colors[10],markersize=3)
# fig.tight_layout()

# # Function to update turbine positions
# def animate(i):
#     line0.set_data(x_all[i]/126., y_all[i]/126.)
#     line1.set_data(range(i+1),flowers_100[0:i+1])
#     line2.set_data(range(i+1),flowers_25[0:i+1])
#     line3.set_data(range(i+1),flowers_5[0:i+1])
#     line4.set_data(range(i+1),park_1[0:i+1])
#     line5.set_data(range(i+1),park_5[0:i+1])
#     line6.set_data(range(i+1),park_10[0:i+1])
#     line7.set_data(range(i+1),gauss_1[0:i+1])
#     line8.set_data(range(i+1),gauss_5[0:i+1])
#     line9.set_data(range(i+1),gauss_10[0:i+1])
#     ax1.legend(['FLOWERS: 100', 'FLOWERS: 25', 'FLOWERS: 5'],loc='upper left')
#     ax2.legend(['Park: $1^\circ$','Park: $5^\circ$','Park: $10^\circ$'],loc='upper left')
#     ax3.legend(['Gauss: $1^\circ$','Gauss: $5^\circ$','Gauss: $10^\circ$'],loc='upper left')
#     return line0, line1, line2, line3, line4, line5, line6, line7, line8, line9

# # Animation
# ani = animation.FuncAnimation(fig, animate, frames=N+1, repeat=False)

# ani.save('mutation.mp4')

# plt.savefig('../mutation.png')
# plt.show()