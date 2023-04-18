import matplotlib.pyplot as plt
import pickle


X, Y, flowers_aep, park_aep, gauss_aep = pickle.load(open('solutions/features_smooth.p','rb'))

X /= 126.
Y /= 126.

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

fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(2,2,1,projection="3d")
ax1 = fig.add_subplot(2,2,2,projection="3d")
# ax2 = fig.add_subplot(2,3,3,projection="3d")
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
# ax5 = fig.add_subplot(2,3,6)
ax0.plot_surface(X,Y,flowers_aep,cmap='viridis')
ax0.set(xlabel='x/D',ylabel='y/D',title='FLOWERS',zticks=[])
ax1.plot_surface(X,Y,park_aep,cmap='viridis')
ax1.set(xlabel='x/D',ylabel='y/D',title='Conventional-Park',zticks=[])
ax3.contour(X,Y,flowers_aep,levels=20,cmap='viridis')
ax4.contour(X,Y,park_aep,levels=20,cmap='viridis')
ax3.set(xlabel='x/D',ylabel='y/D', aspect='equal')
ax4.set(xlabel='x/D',ylabel='y/D', aspect='equal')
fig.tight_layout()

plt.show()