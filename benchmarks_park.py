import numpy as np
import model as set
import tools as tl
import matplotlib.pyplot as plt
import pickle
import visualization as vis
import warnings

save = False

## Case 0
file_name = 'solutions/park' + str(0) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

aep_flowers = np.array(aep_flowers) / aep_flowers[0,:]
aep_floris = np.array(aep_floris) / aep_floris[0,0,:]

time_flowers = np.array(time_flowers) / time_flowers[0]
time_floris = np.array(time_floris) / time_floris[0,0,:]

num_terms = var[0]
num_WD = var[1]
num_WS = var[2]

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232, sharey=ax1)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234, sharex=ax1)
ax5 = fig.add_subplot(235, sharex=ax2, sharey=ax4)
ax6 = fig.add_subplot(236)

ax1.plot(num_terms, np.mean(time_flowers,-1),'-o',markersize=3)
ax1.fill_between(num_terms, np.mean(time_flowers,-1)-np.std(time_flowers,-1),np.mean(time_flowers,-1)+np.std(time_flowers,-1),alpha=0.2)
ax1.set(xlabel='Number of Fourier Terms', ylabel='Normalized Time', title='FLOWERS')

ax2.plot(num_WD, np.mean(time_floris[:,-1],-1),'-o',markersize=3)
ax2.fill_between(num_WD, np.mean(time_floris[:,-1],-1)-np.std(time_floris[:,-1],-1),np.mean(time_floris[:,-1],-1)+np.std(time_floris[:,-1],-1),alpha=0.2)
ax2.set(xlabel='Number of Wind Directions', ylabel='Normalized Time', title='Conventional (Avg)')

ax4.plot(num_terms, np.mean(aep_flowers,-1),'-o',markersize=3)
ax4.fill_between(num_terms, np.mean(aep_flowers,-1)-np.std(aep_flowers,-1),np.mean(aep_flowers,-1)+np.std(aep_flowers,-1),alpha=0.2)
ax4.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

ax5.plot(num_WD, np.mean(aep_floris[:,-1],-1),'-o',markersize=3)
ax5.fill_between(num_WD, np.mean(aep_floris[:,-1],-1)-np.std(aep_floris[:,-1],-1),np.mean(aep_floris[:,-1],-1)+np.std(aep_floris[:,-1],-1),alpha=0.2)
ax5.plot(num_WD, aep_floris[:,-1,0],'-o',markersize=3)
ax5.plot(num_WD, aep_floris[:,-1,1],'-o',markersize=3)
ax5.plot(num_WD, aep_floris[:,-1,2],'-o',markersize=3)
ax5.set(xlabel='Number of Wind Directions', ylabel='Normalized AEP')

wd, ws = np.meshgrid(num_WD,num_WS)
im = ax3.contour(wd,ws,np.mean(time_floris[:,:-1],-1).T,levels=20)
plt.colorbar(im,ax=ax3,label='Normalized Time')
ax3.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds',title='Conventional')
im = ax6.contour(wd,ws,np.mean(aep_floris[:,:-1],-1).T,levels=20)
plt.colorbar(im,ax=ax6,label='Normalized AEP')
ax6.set(xlabel='Number of Wind Directions', ylabel='Number of Wind Speeds')
fig.suptitle('Case 0: Wind Rose Resolution')

# Case 1
file_name = 'solutions/park' + str(1) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

fig, (ax0,ax1) = plt.subplots(1,2,figsize=(11,4))
time_factor = np.zeros_like(time_floris)
for i in range(3):
    time_factor[:,i] = time_floris[:,i] / time_flowers[:,i]
ax0.errorbar(var,np.mean(time_flowers,1),np.std(time_flowers,1),marker='o',ms=3)
ax0.errorbar(var,np.mean(time_floris,1),np.std(time_floris,1),marker='o',ms=3)
ax0.set(xlabel='Number of Turbines',ylabel='Computation Time [s]')
ax0.legend(['FLOWERS','Conventional-Park'])
ax1.errorbar(var,np.mean(time_factor,1),np.std(time_factor,1),marker='o',ms=3,color='green')
ax1.set(xlabel='Number of Turbines',ylabel='Speed-Up Factor')
fig.suptitle('Case 1: Cost Scaling with Number of Turbines')

## Case 2

file_name = 'solutions/park' + str(2) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))
aep_error = [(aep_flowers[i] - aep_floris[i]) / aep_floris[i] * 100 for i in range(len(aep_flowers))]

aep_flow = [aep_flowers[i] / 1e9 for i in range(len(aep_flowers))]
aep_flor = [aep_floris[i] / 1e9 for i in range(len(aep_flowers))]

spacing = var[0]
wr = var[1]

fig, ax = plt.subplots(1,1,figsize=(11,7))
markers = ['o','v','^','s','P','*','X','D','p']
for i in range(9):
    p, cov = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_flor),np.ma.masked_where(wr!=i+1,aep_flow),1,cov=True)
    xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_flor))])
    yrange = p[0]*xrange + p[1]
    # im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_flor),np.ma.masked_where(wr!=i+1,aep_flow),c=wr_val[i]*np.ma.masked_where(wr!=i+1,np.ones(200)),marker=markers[i],vmin=40000.,vmax=82000.,label='WR ' + str(i+1))
    im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_flor),np.ma.masked_where(wr!=i+1,aep_flow),marker=markers[i],label='WR ' + str(i+1))
    ax.plot(xrange,yrange,'--')
# plt.colorbar(im,ax=ax,label='Wind Rose Frequency Standard Deviation')
# xlim = plt.gca().get_xlim()
# ax.plot([0, xlim[1]], [0, xlim[1]], 'k--')
# ax.plot([0, xlim[1]], [0, 0.8*xlim[1]], 'k--')
# ax.fill_between([0, xlim[1]],[0, xlim[1]],[0, 0.8*xlim[1]],alpha=0.2,label='[0%,-20%] Error',color='k')
ax.set(xlabel='FLORIS AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
ax.legend()
fig.suptitle('Case 2: Full Resolution AEP Discrepancy Across Randomized Cases')

## Case 3
freq_std = np.zeros(9)
ws_avg = np.zeros(9)
wr_val = np.zeros(9)
for i in range(9):
    wr = tl.load_wind_rose(i+1)
    freq_std[i] = np.std(wr.freq_val)
    ws_avg[i] = np.sum(wr.freq_val*wr.ws)
    wr_val[i] = ws_avg[i] / (freq_std[i]*360*26)
    # wr = tl.resample_average_ws_by_wd(wr)
    # ws_std[i] = np.mean(wr.ws)

print(wr_val)
file_name = 'solutions/park' + str(3) + '.p'
var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose = pickle.load(open(file_name,'rb'))

aep_flow = [aep_flowers[i] / 1e9 for i in range(len(aep_flowers))]
aep_flor = [aep_floris[i] / 1e9 for i in range(len(aep_flowers))]

spacing = var[0]
wr = var[1]
n_turb = var[2]

fig, (ax0,ax1) = plt.subplots(1,2,figsize=(12,5))

for i in range(9):
    im = ax0.scatter(np.ma.masked_where(wr!=i+1,aep_flor),np.ma.masked_where(wr!=i+1,aep_flow),c=wr_val[i]*np.ma.masked_where(wr!=i+1,np.ones(200)),marker=markers[i],vmin=3.,vmax=9.)
plt.colorbar(im,ax=ax0,label='mean(U)/std(f)')
xlim = ax0.get_xlim()
ax0.plot([0, xlim[1]], [0, xlim[1]], 'k--')
ax0.plot([0, xlim[1]], [0, 0.85*xlim[1]], 'k--')
ax0.fill_between([0, xlim[1]],[0, xlim[1]],[0, 0.85*xlim[1]],alpha=0.2,label='[0%,-15%] Error',color='k')
ax0.legend()
ax0.set(xlabel='FLORIS AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
im = ax1.scatter(time_floris,time_flowers,c=n_turb,cmap='plasma')
plt.colorbar(im,ax=ax1,label='Number of Turbines')
xlim = ax1.get_xlim()
ax1.loglog([10*xlim[0], 10*xlim[1]], xlim, 'k--')
ax1.loglog([50*xlim[0], 50*xlim[1]], xlim, 'k--')
ax1.fill_betweenx(xlim,[10*xlim[0], 10*xlim[1]],[50*xlim[0], 50*xlim[1]],alpha=0.2,label='10x-50x Speed',color='k')
ax1.set(xlabel='FLORIS Time [s]',ylabel='FLOWERS Time [s]')
ax1.legend()
fig.suptitle('Case 3: Low Resolution AEP Discrepancy Across Randomized Cases')

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