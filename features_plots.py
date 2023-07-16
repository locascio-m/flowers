import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.animation as ani
import matplotlib.cm as cm
import matplotlib.colors as co
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
# mov.save('smoothness.mp4')
plt.show()