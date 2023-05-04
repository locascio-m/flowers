import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as ani


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
X, Y, gauss_1 = pickle.load(open('solutions/smooth_gauss1.p','rb'))
_, _, gauss_2 = pickle.load(open('solutions/smooth_gauss2.p','rb'))
_, _, gauss_3 = pickle.load(open('solutions/smooth_gauss3.p','rb'))
_, _, gauss_5 = pickle.load(open('solutions/smooth_gauss5.p','rb'))
_, _, gauss_8 = pickle.load(open('solutions/smooth_gauss8.p','rb'))
_, _, gauss_12 = pickle.load(open('solutions/smooth_gauss12.p','rb'))
_, _, gauss_18 = pickle.load(open('solutions/smooth_gauss18.p','rb'))
_, _, flowers, _, _ = pickle.load(open('solutions/features_smooth.p','rb'))

X /= 126.
Y /= 126.

res = [18,12,8,5,3,2,1]
aep = [gauss_18,gauss_12,gauss_8,gauss_5,gauss_3,gauss_2,gauss_1]

fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].contour(X,Y,flowers,levels=20,cmap='viridis')
ax[0].set(
    title='FLOWERS', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[1].contour(X,Y,gauss_5,levels=20,cmap='viridis')
ax[1].set(
    title='Conventional-Gauss', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)

fig, ax = plt.subplots(1,2,figsize=(8,4))

ax[0].contour(X,Y,flowers,levels=20,cmap='viridis')
ax[0].set(
    title='FLOWERS', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax[1].contour(X,Y,aep[0],levels=20,cmap='viridis')
ax[1].set(
    title='Conventional-Gauss: 18' + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)

def animate(i):
    if i > 0:
        ax[1].clear()
    ax[1].contour(X,Y,aep[i],levels=20,cmap='viridis')
    ax[1].set(
        title='Conventional-Gauss: ' + str(res[i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )

mov = ani.FuncAnimation(fig, animate, interval=1000, frames=7, repeat=True)
mov.save('smoothness.mp4')
plt.show()