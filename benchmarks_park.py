import numpy as np
import model as set
import tools as tl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as co
import pickle
import visualization as vis
import warnings

from scipy.optimize import curve_fit

save = True

# ## Case 0
# file_name = 'solutions/park' + str(0) + '.p'
# var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

# aep_flowers = np.array(aep_flowers) / aep_flowers[0,:]
# aep_floris = np.array(aep_floris) / aep_floris[0,0,:]

# time_flowers = np.array(time_flowers) / time_flowers[0]
# time_floris = np.array(time_floris) / time_floris[0,0,:]

# num_terms = var[0]
# num_WD = var[1]
# num_WS = var[2]

# fig = plt.figure(figsize=(14,7))
# ax1 = fig.add_subplot(231)
# ax2 = fig.add_subplot(232, sharey=ax1)
# ax3 = fig.add_subplot(233)
# ax4 = fig.add_subplot(234, sharex=ax1)
# ax5 = fig.add_subplot(235, sharex=ax2, sharey=ax4)
# ax6 = fig.add_subplot(236)

# ax1.plot(num_terms, np.mean(time_flowers,-1),'-o',markersize=3)
# ax1.fill_between(num_terms, np.mean(time_flowers,-1)-np.std(time_flowers,-1),np.mean(time_flowers,-1)+np.std(time_flowers,-1),alpha=0.2)
# ax1.set(xlabel='Number of Fourier Terms', ylabel='Normalized Time', title='FLOWERS')

# ax2.plot(num_WD, np.mean(time_floris[:,-1],-1),'-o',markersize=3)
# ax2.fill_between(num_WD, np.mean(time_floris[:,-1],-1)-np.std(time_floris[:,-1],-1),np.mean(time_floris[:,-1],-1)+np.std(time_floris[:,-1],-1),alpha=0.2)
# ax2.set(xlabel='Number of Wind Directions', ylabel='Normalized Time', title='Conventional (Avg)')

# ax4.plot(num_terms, np.mean(aep_flowers,-1),'-o',markersize=3)
# ax4.fill_between(num_terms, np.mean(aep_flowers,-1)-np.std(aep_flowers,-1),np.mean(aep_flowers,-1)+np.std(aep_flowers,-1),alpha=0.2)
# ax4.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

# ax5.plot(num_WD, np.mean(aep_floris[:,-1],-1),'-o',markersize=3)
# ax5.fill_between(num_WD, np.mean(aep_floris[:,-1],-1)-np.std(aep_floris[:,-1],-1),np.mean(aep_floris[:,-1],-1)+np.std(aep_floris[:,-1],-1),alpha=0.2)
# ax5.set(xlabel='Number of Wind Directions', ylabel='Normalized AEP')

# wd, ws = np.meshgrid(num_WD,num_WS)
# im = ax3.contour(wd,ws,np.mean(time_floris[:,:-1],-1).T,levels=20)
# plt.colorbar(im,ax=ax3,label='Normalized Time')
# ax3.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds',title='Conventional')
# im = ax6.contour(wd,ws,np.mean(aep_floris[:,:-1],-1).T,levels=20)
# plt.colorbar(im,ax=ax6,label='Normalized AEP')
# ax6.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds')
# fig.suptitle('Case 0: Wind Rose Resolution')

## Tuning: Option 1
file_name = 'solutions/park' + str(0) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

aep_flowers = np.array(aep_flowers) / aep_flowers[0,:]
aep_floris = np.array(aep_floris) / aep_floris[0,0,:]

time_flowers = np.array(time_flowers) / time_flowers[0]
time_floris = np.array(time_floris) / time_floris[0,0,:]

num_terms = var[0]
num_WD = var[1]
num_WS = var[2]

wd, ws = np.meshgrid(num_WD,num_WS)

cmap = cm.get_cmap('viridis')

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(222)
ax6 = fig.add_subplot(224)
ax4 = fig.add_subplot(223,sharey=ax1)

ax1.plot(num_terms, np.mean(time_flowers,-1),'-o',markersize=3)
# ax1.fill_between(num_terms, np.mean(time_flowers,-1)-np.std(time_flowers,-1),np.mean(time_flowers,-1)+np.std(time_flowers,-1),alpha=0.2)
ax1.set(ylabel='Normalized Time', title='FLOWERS')

ax4.plot(num_terms, np.mean(aep_flowers,-1),'-o',markersize=3)
# ax4.fill_between(num_terms, np.mean(aep_flowers,-1)-np.std(aep_flowers,-1),np.mean(aep_flowers,-1)+np.std(aep_flowers,-1),alpha=0.2)
ax4.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

im = ax3.contourf(wd,ws,np.mean(time_floris[:,:-1],-1).T,levels=20)
plt.colorbar(im,ax=ax3,label='Normalized Time')
ax3.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds',title='Conventional')
im = ax6.contourf(wd,ws,np.mean(aep_floris[:,:-1],-1).T,levels=20)
plt.colorbar(im,ax=ax6,label='Normalized AEP')
ax6.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds')
if save == True:
    plt.savefig('../tuning.png')

# for i in range(len(num_WS)):
#     ax3.plot(num_WD,np.mean(time_floris[:,i,:],-1),'-o',markersize=3,color=cmap(num_WS[i]/26.0001))
#     ax3.fill_between(num_WD, np.mean(time_floris[:,i,:],-1)-np.std(time_floris[:,i],-1),np.mean(time_floris[:,i],-1)+np.std(time_floris[:,i],-1),alpha=0.2,color=cmap(num_WS[i]/26.0001))
# cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=3,vmax=26.0001), cmap=cmap),ax=ax3,label='Number of Wind Speeds')
# cbar.ax.set_yticks([3,4,5,7,10,15,20,26])
# ax3.set(title='Conventional')
# for i in range(len(num_WS)):
#     ax6.plot(num_WD,np.mean(aep_floris[:,i,:],-1),'-o',markersize=3,color=cmap(num_WS[i]/26.0001))
#     ax6.fill_between(num_WD, np.mean(aep_floris[:,i,:],-1)-np.std(aep_floris[:,i],-1),np.mean(aep_floris[:,i],-1)+np.std(aep_floris[:,i],-1),alpha=0.2,color=cmap(num_WS[i]/26.0001))
# cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=3,vmax=26.0001), cmap=cmap),ax=ax6,label='Number of Wind Speeds')
# cbar.ax.set_yticks([3,4,5,7,10,15,20,26])
# # cbar.ax.set_yticklabels(num_WS)
# ax6.set(xlabel='Number of Wind Directions')

## Tuning: Option 2
file_name = 'solutions/park' + str(0) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

aep_flowers = np.array(aep_flowers) / aep_flowers[0,:]
aep_floris = np.array(aep_floris) / aep_floris[0,0,:]

time_flowers = np.array(time_flowers) / time_flowers[0]
time_floris = np.array(time_floris) / time_floris[0,0,:]

num_terms = var[0]
num_WD = var[1]
num_WS = var[2]

cmap = cm.get_cmap('viridis')

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(222, sharey=ax1)
ax6 = fig.add_subplot(224, sharex=ax3)
ax4 = fig.add_subplot(223, sharex=ax1, sharey=ax6)

ax1.plot(num_terms, np.mean(time_flowers,-1),'-o',markersize=3)
ax1.fill_between(num_terms, np.mean(time_flowers,-1)-np.std(time_flowers,-1),np.mean(time_flowers,-1)+np.std(time_flowers,-1),alpha=0.2)
ax1.set(ylabel='Normalized Time', title='FLOWERS')

ax4.plot(num_terms, np.mean(aep_flowers,-1),'-o',markersize=3)
ax4.fill_between(num_terms, np.mean(aep_flowers,-1)-np.std(aep_flowers,-1),np.mean(aep_flowers,-1)+np.std(aep_flowers,-1),alpha=0.2)
ax4.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

for i in range(len(num_WS)):
    ax3.plot(num_WD,np.mean(time_floris[:,i,:],-1),'-o',markersize=3,color=cmap(num_WS[i]/26.0001))
    ax3.fill_between(num_WD, np.mean(time_floris[:,i,:],-1)-np.std(time_floris[:,i],-1),np.mean(time_floris[:,i],-1)+np.std(time_floris[:,i],-1),alpha=0.2,color=cmap(num_WS[i]/26.0001))
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=3,vmax=26.0001), cmap=cmap),ax=ax3,label='Number of Wind Speeds')
cbar.ax.set_yticks([3,4,5,7,10,15,20,26])
ax3.set(title='Conventional')
for i in range(len(num_WS)):
    ax6.plot(num_WD,np.mean(aep_floris[:,i,:],-1),'-o',markersize=3,color=cmap(num_WS[i]/26.0001))
    ax6.fill_between(num_WD, np.mean(aep_floris[:,i,:],-1)-np.std(aep_floris[:,i],-1),np.mean(aep_floris[:,i],-1)+np.std(aep_floris[:,i],-1),alpha=0.2,color=cmap(num_WS[i]/26.0001))
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=3,vmax=26.0001), cmap=cmap),ax=ax6,label='Number of Wind Speeds')
cbar.ax.set_yticks([3,4,5,7,10,15,20,26])
# cbar.ax.set_yticklabels(num_WS)
ax6.set(xlabel='Number of Wind Directions')

# wd, ws = np.meshgrid(num_WD,num_WS)
# im = ax6.contour(wd,ws,np.mean(aep_floris[:,:-1],-1).T,levels=20)
# plt.colorbar(im,ax=ax6,label='Normalized AEP')
# ax6.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds')
# fig.suptitle('Case 0: Wind Rose Resolution')

# # Case 1
# file_name = 'solutions/park' + str(1) + '.p'
# var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

# fig, (ax0,ax1) = plt.subplots(1,2,figsize=(11,4))
# time_factor = np.zeros_like(time_floris)
# for i in range(3):
#     time_factor[:,i] = time_floris[:,i] / time_flowers[:,i]
# ax0.errorbar(var,np.mean(time_flowers,1),np.std(time_flowers,1),marker='o',ms=3)
# ax0.errorbar(var,np.mean(time_floris,1),np.std(time_floris,1),marker='o',ms=3)
# ax0.set(xlabel='Number of Turbines',ylabel='Computation Time [s]')
# ax0.legend(['FLOWERS','Conventional-Park'])
# ax1.errorbar(var,np.mean(time_factor,1),np.std(time_factor,1),marker='o',ms=3,color='green')
# ax1.set(xlabel='Number of Turbines',ylabel='Speed-Up Factor')
# fig.suptitle('Case 1: Cost Scaling with Number of Turbines')

## Case 2

file_name = 'solutions/park' + str(2) + '.p'
var, aep_flowers_full, aep_flowers_fast, aep_park_full, aep_park_fast, time_flowers_full, time_flowers_fast, time_park_full, time_park_fast = pickle.load(open(file_name,'rb'))

# aep_error = [(aep_flowers[i] - aep_floris[i]) / aep_floris[i] * 100 for i in range(len(aep_flowers))]

aep_flowers_full = np.array(aep_flowers_full) / 1e9
aep_flowers_fast = np.array(aep_flowers_fast) / 1e9
aep_park_full = np.array(aep_park_full) / 1e9
aep_park_fast = np.array(aep_park_fast) / 1e9

wr = var[0]
nt = var[1]
cmap = cm.get_cmap('coolwarm')

fig, ax = plt.subplots(1,1,figsize=(11,7))
markers = ['o','v','^','s','P','*','X','D','p']
for i in range(9):
    p, _ = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park_full),np.ma.masked_where(wr!=i+1,aep_flowers_full),1,cov=True)
    xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park_full))])
    yrange = p[0]*xrange + p[1]
    im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_park_full),np.ma.masked_where(wr!=i+1,aep_flowers_full),marker=markers[i],label='WR ' + str(i+1))
    ax.plot(xrange,yrange,'--')
xlim = ax.get_xlim()
ax.fill_between([0, xlim[1]],[0, xlim[1]],[0, 0.7*xlim[1]],alpha=0.2,label='[-30%,0%]',color='k')
# ax.plot([0, xlim[1]], [0, xlim[1]], 'k--')
ax.set(xlabel='Conventional AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
ax.legend()
if save == True:
    plt.savefig('../full_aep.png')

# Fast AEP Comparison
cmap = cm.get_cmap('coolwarm')

fig, ax = plt.subplots(1,2,figsize=(14,5))
ax[0].scatter(aep_park_full, aep_flowers_fast,10, zorder=9)
xlim = ax[0].get_xlim()
ax[0].plot([0,xlim[1]],[0,xlim[1]],'k--',linewidth=2, zorder=10)
ax[0].fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1))
ax[0].fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2))
ax[0].fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3))
ax[0].fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 0.99*xlim[1]],alpha=0.4,label='[-5%,-1%]',color=cmap(0.4))
# ax[0].fill_between([0, xlim[1]],[0, 0.99*xlim[1]],[0, 1.01*xlim[1]],alpha=0.4,label='$\pm$1%',color=cmap(0.5))
ax[0].fill_between([0, xlim[1]],[0, 1.01*xlim[1]],[0, 1.05*xlim[1]],alpha=0.4,label='[1%,5%]',color=cmap(0.6))
ax[0].fill_between([0, xlim[1]],[0, 1.05*xlim[1]],[0, 1.1*xlim[1]],alpha=0.4,label='[5%,10%]',color=cmap(0.7))
ax[0].fill_between([0, xlim[1]],[0, 1.1*xlim[1]],[0, 1.2*xlim[1]],alpha=0.4,label='[10%,20%]',color=cmap(0.8))
ax[0].fill_between([0, xlim[1]],[0, 1.2*xlim[1]],[0, 1.3*xlim[1]],alpha=0.4,label='[20%,30%]',color=cmap(0.9))
ax[0].legend()
ax[0].set(xlabel='Conventional AEP [GWh]',ylabel='FLOWERS AEP [GWh]')

ax[1].scatter(aep_park_full, aep_park_fast,10, color='tab:orange',zorder=9)
ax[1].plot([0,xlim[1]],[0,xlim[1]],'k--',linewidth=2, zorder=10)
ax[1].fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1))
ax[1].fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2))
ax[1].fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3))
ax[1].fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 0.99*xlim[1]],alpha=0.4,label='[-5%,-1%]',color=cmap(0.4))
# ax[0].fill_between([0, xlim[1]],[0, 0.99*xlim[1]],[0, 1.01*xlim[1]],alpha=0.4,label='$\pm$1%',color=cmap(0.5))
ax[1].fill_between([0, xlim[1]],[0, 1.01*xlim[1]],[0, 1.05*xlim[1]],alpha=0.4,label='[1%,5%]',color=cmap(0.6))
ax[1].fill_between([0, xlim[1]],[0, 1.05*xlim[1]],[0, 1.1*xlim[1]],alpha=0.4,label='[5%,10%]',color=cmap(0.7))
ax[1].fill_between([0, xlim[1]],[0, 1.1*xlim[1]],[0, 1.2*xlim[1]],alpha=0.4,label='[10%,20%]',color=cmap(0.8))
ax[1].fill_between([0, xlim[1]],[0, 1.2*xlim[1]],[0, 1.3*xlim[1]],alpha=0.4,label='[20%,30%]',color=cmap(0.9))
ax[1].set(xlabel='Conventional AEP [GWh]',ylabel='Conventional-Fast AEP [GWh]')

if save == True:
    plt.savefig('../fast_aep.png')

## Cost comparison

cmap = cm.get_cmap('Greens')

fig, ax = plt.subplots(1,1,figsize=(11,7))
im = ax.scatter(time_park_fast,time_flowers_fast,c=nt,cmap='plasma',zorder=5)
ax.set(xscale='log',yscale='log')
plt.autoscale(False)
plt.colorbar(im,ax=ax,label='Number of Turbines')
# xlim = ax.get_xlim()
# ylim = ax.get_xlim()
# ax.loglog([xlim[0], xlim[1]], xlim, 'k--')
# ax.loglog([10*xlim[0], 10*xlim[1]], xlim, 'k--')
# ax.loglog([20*xlim[0], 20*xlim[1]], xlim, 'k--')
# ax.loglog([50*xlim[0], 50*xlim[1]], xlim, 'k--')
# ax.loglog([100*xlim[0], 100*xlim[1]], xlim, 'k--')
ax.fill_betweenx(xlim,[xlim[0], xlim[1]],[10*xlim[0], 10*xlim[1]],alpha=0.4,label='1x-10x Speed',color=cmap(0.2))
ax.fill_betweenx(xlim,[10*xlim[0], 10*xlim[1]],[20*xlim[0], 20*xlim[1]],alpha=0.4,label='10x-20x Speed',color=cmap(0.4))
ax.fill_betweenx(xlim,[20*xlim[0], 20*xlim[1]],[50*xlim[0], 50*xlim[1]],alpha=0.4,label='20x-50x Speed',color=cmap(0.6))
ax.fill_betweenx(xlim,[50*xlim[0], 50*xlim[1]],[100*xlim[0], 100*xlim[1]],alpha=0.4,label='50x-100x Speed',color=cmap(0.999))

ax.set(xlabel='Conventional-Fast Time [s]',ylabel='FLOWERS Time [s]')
ax.legend()
if save == True:
    plt.savefig('../cost.png')

## Cost scaling

def power_law(x, a, b):
    return a*np.power(x, b)

p , _ = curve_fit(power_law,nt,time_flowers_fast)
q , _ = curve_fit(power_law,nt,time_park_fast)

fig, ax = plt.subplots(1,1,figsize=(11,7))
ax.scatter(nt,time_flowers_fast,s=40,facecolors='none', edgecolors='tab:blue')
ax.scatter(nt,time_park_fast,s=40,facecolors='none', edgecolors='tab:orange')
ax.plot(range(1,101),power_law(range(1,101),p[0],p[1]),'--',linewidth=3,label='FLOWERS: $\mathcal{O}(N^{1.28})$')
ax.plot(range(1,101),power_law(range(1,101),q[0],q[1]),'--',linewidth=3,label='Conventional: $\mathcal{O}(N^{1.26})$')
ax.set(xlabel='Number of Turbines',ylabel='AEP Evaluation Time [s]')
ax.legend()

if save == True:
    plt.savefig('../cost_scaling.png')

# ## Case 3
# freq_std = np.zeros(9)
# ws_avg = np.zeros(9)
# wr_val = np.zeros(9)
# for i in range(9):
#     wr = tl.load_wind_rose(i+1)
#     freq_std[i] = np.std(wr.freq_val)
#     ws_avg[i] = np.sum(wr.freq_val*wr.ws)
#     wr_val[i] = ws_avg[i] / (freq_std[i]*360*26)
#     # wr = tl.resample_average_ws_by_wd(wr)
#     # ws_std[i] = np.mean(wr.ws)

# file_name = 'solutions/park' + str(3) + '.p'
# var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

# aep_flow = [aep_flowers[i] / 1e9 for i in range(len(aep_flowers))]
# aep_flor = [aep_floris[i] / 1e9 for i in range(len(aep_flowers))]

# spacing = var[0]
# wr = var[1]
# n_turb = var[2]

# fig, (ax0,ax1) = plt.subplots(1,2,figsize=(12,5))

# for i in range(9):
#     im = ax0.scatter(np.ma.masked_where(wr!=i+1,aep_flor),np.ma.masked_where(wr!=i+1,aep_flow),c=wr_val[i]*np.ma.masked_where(wr!=i+1,np.ones(200)),marker=markers[i],vmin=3.,vmax=9.)
# plt.colorbar(im,ax=ax0,label='mean($U_\infty$) / std($f_{u_\infty,\phi}$)')
# xlim = ax0.get_xlim()
# ax0.plot([0, xlim[1]], [0, xlim[1]], 'k--')
# ax0.plot([0, xlim[1]], [0, 0.8*xlim[1]], 'k--')
# ax0.fill_between([0, xlim[1]],[0, xlim[1]],[0, 0.8*xlim[1]],alpha=0.2,label='[0%,-20%] Error',color='k')
# ax0.legend()
# ax0.set(xlabel='Conventional AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
# im = ax1.scatter(time_floris,time_flowers,c=n_turb,cmap='plasma')
# plt.colorbar(im,ax=ax1,label='Number of Turbines')
# xlim = ax1.get_xlim()
# ax1.loglog([10*xlim[0], 10*xlim[1]], xlim, 'k--')
# ax1.loglog([50*xlim[0], 50*xlim[1]], xlim, 'k--')
# ax1.fill_betweenx(xlim,[10*xlim[0], 10*xlim[1]],[50*xlim[0], 50*xlim[1]],alpha=0.2,label='10x-50x Speed',color='k')
# ax1.set(xlabel='Conventional Time [s]',ylabel='FLOWERS Time [s]')
# ax1.legend()
# fig.suptitle('Case 3: Low Resolution AEP Discrepancy Across Randomized Cases')

# for case in [0]:

#     file_name = 'solutions/bench' + str(case) + '.p'

#     if case == 6:
#         var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose, TI = pickle.load(open(file_name,'rb'))

#         aep_flowers = np.array(aep_flowers) / aep_flowers[0]
#         aep_floris = np.array(aep_floris) / aep_floris[0]

#         time_flowers = np.array(time_flowers) / time_flowers[0]
#         time_floris = np.array(time_floris) / time_floris[0]

#         num_terms = var[0]
#         num_bins = var[1]

#         # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2,figsize = (11,7))

#         fig = plt.figure(figsize=(14,7))
#         ax0 = fig.add_subplot(231)
#         ax1 = fig.add_subplot(232, sharey=ax0)
#         ax2 = fig.add_subplot(234, sharex=ax0)
#         ax3 = fig.add_subplot(235, sharex=ax1, sharey=ax2)
#         ax4 = fig.add_subplot(233, polar=True)
#         ax5 = fig.add_subplot(236)

#         ax0.plot(num_terms, time_flowers,'-o',markersize=3)
#         ax0.set(xlabel='Number of Fourier Terms', ylabel='Normalized Time', title='FLOWERS')

#         ax1.plot(num_bins, time_floris,'-o',markersize=3)
#         ax1.set(xlabel='Number of Wind Directions', ylabel='Normalized Time', title='Conventional')

#         ax2.plot(num_terms, aep_flowers,'-o',markersize=3)
#         ax2.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

#         ax3.plot(num_bins, aep_floris,'-o',markersize=3)
#         ax3.set(xlabel='Number of Wind Directions', ylabel='Normalized AEP')

#         fig.suptitle(titles[case])

#     else:
        
#         var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose, TI = pickle.load(open(file_name,'rb'))

#         # if case == 4 or case == 5:
#         #     tmp = [str(elem) for elem in var]
#         #     var = range(len(tmp))

#         aep_error = [(aep_flowers[i] - aep_floris[i]) / aep_floris[i] * 100 for i in range(len(aep_flowers))]
#         time_factor = [time_floris[i] / time_flowers[i] for i in range(len(aep_flowers))]

#         fig = plt.figure(figsize=(14,7))
#         ax0 = fig.add_subplot(231)
#         ax1 = fig.add_subplot(232)
#         ax2 = fig.add_subplot(234)
#         ax3 = fig.add_subplot(235)
#         ax4 = fig.add_subplot(233, polar=True)
#         ax5 = fig.add_subplot(236)

#         ax0.plot(var,[elem / 1e9 for elem in aep_flowers],'-o',markersize=3)
#         ax0.plot(var,[elem / 1e9 for elem in aep_floris],'-o',markersize=3)
#         ax0.set(xlabel=xlabels[case], ylabel='AEP [GWh]')
#         # if case == 4 or case == 5:
#         #     ax0.set(xticklabels=tmp)
#         ax0.legend(['FLOWERS','Conventional'])

#         ax1.plot(var,time_flowers,'-o',markersize=3)
#         ax1.plot(var,time_floris,'-o',markersize=3)
#         ax1.set(xlabel=xlabels[case], ylabel='Time [s]')
#         # if case == 4 or case == 5:
#         #     ax1.set(xticklabels=tmp)

#         ax2.plot(var, aep_error, 'g-o', markersize=3)
#         ax2.set(xlabel=xlabels[case], ylabel='AEP Difference [%]')
#         # if case == 4 or case == 5:
#         #     ax2.set(xticklabels=tmp)

#         ax3.plot(var, time_factor, 'g-o', markersize=3)
#         ax3.set(xlabel=xlabels[case], ylabel='Time Factor [x]')
#         # if case == 4 or case == 5:
#         #     ax3.set(xticklabels=tmp)
#         fig.suptitle(titles[case])

#     vis.plot_wind_rose(wind_rose, ax=ax4)
#     vis.plot_layout(layout_x, layout_y, ax=ax5)
#     ax5.set(title='TI = {:.2f}'.format(TI))
#     fig.tight_layout()
#     if save:
#         plt.savefig("/Users/locascio/Library/Mobile Documents/com~apple~CloudDocs/Research/FLOWERS Improvements/case_calibration" + str(case), dpi=500)

plt.show()