import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.animation as ani
import matplotlib.cm as cm
import matplotlib.colors as co
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

save = True

# X, Y, flowers_aep, park_aep, gauss_aep = pickle.load(open('solutions/features_smooth.p','rb'))

# X /= 126.
# Y /= 126.

# Surface plots of normalized AEP
# fig = plt.figure(figsize=(11,7))
# ax0 = fig.add_subplot(2,3,1,projection="3d")
# ax1 = fig.add_subplot(2,3,2,projection="3d")
# ax2 = fig.add_subplot(2,3,3,projection="3d")
# ax3 = fig.add_subplot(2,3,4)
# ax4 = fig.add_subplot(2,3,5)
# ax5 = fig.add_subplot(2,3,6)
# ax0.plot_surface(X,Y,flowers_aep,cmap='viridis')
# ax0.set(xlabel='x [m]',ylabel='y [m]',title='FLOWERS')
# ax1.plot_surface(X,Y,park_aep,cmap='viridis')
# ax1.set(xlabel='x [m]',ylabel='y [m]',title='Conventional-Park')
# ax2.plot_surface(X,Y,gauss_aep,cmap='viridis')
# ax2.set(xlabel='x [m]',ylabel='y [m]',title='Conventional-Gauss')
# ax3.contour(X,Y,flowers_aep,levels=20,cmap='viridis')
# ax4.contour(X,Y,park_aep,levels=20,cmap='viridis')
# ax5.contour(X,Y,gauss_aep,levels=20,cmap='viridis')
# ax3.set(xlabel='x [m]',ylabel='y [m]', aspect='equal')
# ax4.set(xlabel='x [m]',ylabel='y [m]', aspect='equal')
# ax5.set(xlabel='x [m]',ylabel='y [m]', aspect='equal')

# fig = plt.figure(figsize=(11,7))
# ax0 = fig.add_subplot(2,2,1,projection="3d")
# ax1 = fig.add_subplot(2,2,2,projection="3d")
# # ax2 = fig.add_subplot(2,3,3,projection="3d")
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)
# # ax5 = fig.add_subplot(2,3,6)
# ax0.plot_surface(X,Y,flowers_aep,cmap='viridis')
# ax0.set(xlabel='x/D',ylabel='y/D',title='FLOWERS',zticks=[])
# ax1.plot_surface(X,Y,park_aep,cmap='viridis')
# ax1.set(xlabel='x/D',ylabel='y/D',title='Conventional-Park',zticks=[])
# ax3.contour(X,Y,flowers_aep,levels=20,cmap='viridis')
# ax4.contour(X,Y,park_aep,levels=20,cmap='viridis')
# ax3.set(xlabel='x/D',ylabel='y/D', aspect='equal')
# ax4.set(xlabel='x/D',ylabel='y/D', aspect='equal')
# fig.tight_layout()

# Animation
res = [18,15,12,10,8,6,5,4,3,2,1]
terms = [3,4,5,10,15,20,30,45,60,90,180]
aep_park = []
aep_gauss = []
aep_flowers = []

for idx in range(len(res)):
    file_name = 'solutions/smooth_park' + str(res[idx]) + '.p'
    X, Y, aep = pickle.load(open(file_name,'rb'))
    aep /= np.max(aep)
    aep_park.append(aep)

    file_name = 'solutions/smooth_gauss' + str(res[idx]) + '.p'
    X, Y, aep = pickle.load(open(file_name,'rb'))
    aep /= np.max(aep)
    aep_gauss.append(aep)

    file_name = 'solutions/smooth_flowers' + str(terms[idx]) + '.p'
    _, _, aep = pickle.load(open(file_name,'rb'))
    aep /= np.max(aep)
    aep_flowers.append(aep)


X /= 126.
Y /= 126.

fig, ax = plt.subplots(1,3,figsize=(10,4))
divider = make_axes_locatable(ax[-1])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0].contour(X,Y,aep_flowers[2],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[0].set(
    title='FLOWERS', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[1].contour(X,Y,aep_park[6],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[1].set(
    title='Conventional-Park', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[2].contour(X,Y,aep_gauss[6],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[2].set(
    title='Conventional-Gauss', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.95,vmax=1.)),cax=cax,label='Normalized AEP')
if save == True:
    plt.savefig('../smoothness.png')

fig, ax = plt.subplots(1,3,figsize=(10,4))
divider = make_axes_locatable(ax[-1])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0].contour(X,Y,aep_flowers[0],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[0].set(
    title='FLOWERS: 3', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[1].contour(X,Y,aep_park[0],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[1].set(
    title='Conventional-Park: 18' + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[2].contour(X,Y,aep_gauss[0],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax[2].set(
    title='Conventional-Gauss: 18' + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.95,vmax=1.)),cax=cax,label='Normalized AEP')

def animate(i):
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    
    ax[0].contour(X,Y,aep_flowers[i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax[0].set(
        title='FLOWERS: ' + str(terms[i]), 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax[1].contour(X,Y,aep_park[i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax[1].set(
        title='Conventional-Park: ' + str(res[i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax[2].contour(X,Y,aep_gauss[i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax[2].set(
        title='Conventional-Gauss: ' + str(res[i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )

mov = ani.FuncAnimation(fig, animate, interval=1000, frames=len(res), repeat=True)
if save == True:
    mov.save('smoothness.mp4')

# Mutation

x_all, y_all, flowers, park, gauss = pickle.load(open('solutions/mutation.p','rb'))

flowers_100 = flowers[0]
flowers_25 = flowers[1]
flowers_5 = flowers[2]
park_1 = park[0]
park_5 = park[1]
park_10 = park[2]
gauss_1 = gauss[0]
gauss_5 = gauss[1]
gauss_10 = gauss[2]
N = len(x_all) - 1

fig = plt.figure(figsize=(12,7))
ax0 = plt.subplot2grid((3,2),(0,0),rowspan=3)
ax1 = plt.subplot2grid((3,2),(0,1))
ax2 = plt.subplot2grid((3,2),(1,1))
ax3 = plt.subplot2grid((3,2),(2,1))

# fig, (ax0,ax1) = plt.subplots(1,2, )
ax0.set(aspect='equal', xlim=[-5,35], ylim=[-5,45], xlabel='x/D', ylabel='y/D')
ax1.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax2.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax3.set(xlim=[0,N], ylim=[0.975,1.025], xlabel='Iteration', ylabel='Normalized AEP')

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

line0, = ax0.plot([],[],"o",color='tab:red',markersize=7)
line1, = ax1.plot([],[],"-o",color=colors[0],markersize=3)
line2, = ax1.plot([],[],":o",color=colors[1],markersize=3)
line3, = ax1.plot([],[],"--o",color=colors[2],markersize=3)
line4, = ax2.plot([],[],"-o",color=colors[4],markersize=3)
line5, = ax2.plot([],[],":o",color=colors[5],markersize=3)
line6, = ax2.plot([],[],"--o",color=colors[6],markersize=3)
line7, = ax3.plot([],[],"-o",color=colors[8],markersize=3)
line8, = ax3.plot([],[],":o",color=colors[9],markersize=3)
line9, = ax3.plot([],[],"--o",color=colors[10],markersize=3)
fig.tight_layout()

# Function to update turbine positions
def animate(i):
    line0.set_data(x_all[i]/126., y_all[i]/126.)
    line1.set_data(range(i+1),flowers_100[0:i+1])
    line2.set_data(range(i+1),flowers_25[0:i+1])
    line3.set_data(range(i+1),flowers_5[0:i+1])
    line4.set_data(range(i+1),park_1[0:i+1])
    line5.set_data(range(i+1),park_5[0:i+1])
    line6.set_data(range(i+1),park_10[0:i+1])
    line7.set_data(range(i+1),gauss_1[0:i+1])
    line8.set_data(range(i+1),gauss_5[0:i+1])
    line9.set_data(range(i+1),gauss_10[0:i+1])
    ax1.legend(['FLOWERS: 100', 'FLOWERS: 25', 'FLOWERS: 5'],loc='upper left')
    ax2.legend(['Park: $1^\circ$','Park: $5^\circ$','Park: $10^\circ$'],loc='upper left')
    ax3.legend(['Gauss: $1^\circ$','Gauss: $5^\circ$','Gauss: $10^\circ$'],loc='upper left')
    return line0, line1, line2, line3, line4, line5, line6, line7, line8, line9

# Animation
ani = animation.FuncAnimation(fig, animate, frames=N+1, repeat=False)

if save == True:
    ani.save('mutation.mp4')
    plt.savefig('../mutation.png')

plt.show()