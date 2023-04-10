import matplotlib.pyplot as plt
import pickle


X, Y, flowers_aep, floris_aep = pickle.load(open('solutions/features_aep.p','rb'))

# Surface plots of normalized AEP
fig, (ax0,ax1) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
ax0.plot_surface(X,Y,flowers_aep,cmap='viridis')
ax0.set(xlabel='x [m]',ylabel='y [m]',title='FLOWERS')
ax1.plot_surface(X,Y,floris_aep,cmap='viridis')
ax1.set(xlabel='x [m]',ylabel='y [m]',title='Conventional-Park')

plt.show()