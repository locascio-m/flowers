import numpy as np
import tools as tl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as co
import pickle
import visualization as vis
from scipy.optimize import curve_fit
import matplotlib.animation as animation
import matplotlib.collections as coll
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

save = False
dpi = 500

###########################################################################
# RANDOMIZED LAYOUT / WIND ROSE
###########################################################################
var, aep_flowers, aep_park, time_flowers, time_park = pickle.load(open('solutions/aep0.p','rb'))
wr = var[0]
nt = var[1]

aep_flowers /= 1e9
aep_park /= 1e9

# Remove outlier
idx = np.argmax(time_park)
wr = np.delete(wr,idx)
nt = np.delete(nt,idx)
aep_flowers = np.delete(aep_flowers,idx)
aep_park = np.delete(aep_park,idx)
time_flowers = np.delete(time_flowers,idx)
time_park = np.delete(time_park,idx)


# AEP Comparison
fig, ax = plt.subplots(1,1,figsize=(7,5))
markers = ['o','v','^','s','P','*','X','D','p']
cmap2 = cm.get_cmap('hot')
for i in range(9):
    p = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),1)
    xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park))])
    yrange = p[0]*xrange + p[1]
    im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),20,alpha=0.90,marker=markers[i],zorder=15,color=cmap2(0.1*i))
    ax.plot([],[],linestyle='--',markersize=5,marker=markers[i],label=str(i+1),color=cmap2(0.1*i))
    ax.plot(xrange,yrange,'--',zorder=14,color=cmap2(0.1*i))
xlim = ax.get_xlim()
cmap = cm.get_cmap('coolwarm_r')
ax.plot([0,xlim[1]],[0,xlim[1]],'k',linewidth=2, label='0%',zorder=10)
ax.fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 1*xlim[1]],alpha=0.4,label='[-5%,0%]',color=cmap(0.4),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1),zorder=1)
ax.set(xlabel='Conventional-Park AEP [GWh]',ylabel='FLOWERS AEP [GWh]',aspect='equal')
handles, labels = ax.get_legend_handles_labels()
tmp = ax.legend(handles[9:],labels[9:],bbox_to_anchor=(0.215,0.1099999,0.9,0.89),title='AEP % Diff.')
ax.add_artist(tmp)
tmp2 = ax.legend(handles[0:9],labels[0:9],loc='upper left',title='Wind Rose')
tmp2.get_frame().set_facecolor('gainsboro')
axins = zoomed_inset_axes(ax, zoom=6, loc='lower right')
for i in range(9):
    p = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),1)
    xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park))])
    yrange = p[0]*xrange + p[1]
    im = axins.scatter(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),20,alpha=0.90,marker=markers[i],zorder=15,color=cmap2(0.1*i))
    axins.plot([],[],linestyle='--',markersize=5,marker=markers[i],label=str(i+1),color=cmap2(0.1*i))
    axins.plot(xrange,yrange,'--',zorder=14,color=cmap2(0.1*i))
xlim = axins.get_xlim()
cmap = cm.get_cmap('coolwarm_r')
axins.plot([0,xlim[1]],[0,xlim[1]],'k',linewidth=2, label='0%',zorder=10)
axins.fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 1*xlim[1]],alpha=0.4,label='[-5%,0%]',color=cmap(0.4),zorder=1)
axins.fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3),zorder=1)
axins.fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2),zorder=1)
axins.fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1),zorder=1)
axins.set(xlim=[200,1200],ylim=[200,1200],aspect='equal',xticks=[],yticks=[])
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", zorder=25)

# for i in range(9):
#     p = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),1)
#     xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park))])
#     yrange = p[0]*xrange + p[1]
#     im = ax[1].scatter(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),20,alpha=0.90,marker=markers[i],zorder=15,color=cmap2(0.1*i))
#     ax[1].plot([],[],linestyle='--',markersize=5,marker=markers[i],label=str(i+1),color=cmap2(0.1*i))
#     ax[1].plot(xrange,yrange,'--',zorder=14,color=cmap2(0.1*i))
# xlim = ax[1].get_xlim()
# cmap = cm.get_cmap('coolwarm_r')
# ax[1].plot([0,xlim[1]],[0,xlim[1]],'k',linewidth=2, label='0%',zorder=10)
# ax[1].fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 0.99*xlim[1]],alpha=0.4,label='[-5%,-1%]',color=cmap(0.4),zorder=1)
# ax[1].fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3),zorder=1)
# ax[1].fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2),zorder=1)
# ax[1].fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1),zorder=1)
# ax[1].set(xlabel='Conventional-Park AEP [GWh]',ylabel='FLOWERS AEP [GWh]',xlim=[500,2500],ylim=[500,2500],aspect='equal',xticks=[500,1000,1500,2000,2500],yticks=[500,1000,1500,2000,2500])
fig.tight_layout()
if save:
    plt.savefig('./figures/aep_comparison.png', dpi=dpi)
# fig, ax = plt.subplots(1,1)
# markers = ['o','v','^','s','P','*','X','D','p']
# cmap2 = cm.get_cmap('hot')
# for i in range(9):
#     p = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),1)
#     xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park))])
#     yrange = p[0]*xrange + p[1]
#     im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),30,alpha=0.85,marker=markers[i],zorder=15,color=cmap2(0.1*i))
#     ax.plot([],[],linestyle='--',markersize=5,marker=markers[i],label=str(i+1),color=cmap2(0.1*i))
#     ax.plot(xrange,yrange,'--',zorder=14,color=cmap2(0.1*i))
# xlim = ax.get_xlim()
# cmap = cm.get_cmap('coolwarm_r')
# ax.plot([0,xlim[1]],[0,xlim[1]],'k',linewidth=2, label='0%',zorder=10)
# ax.fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 0.99*xlim[1]],alpha=0.4,label='[-5%,-1%]',color=cmap(0.4),zorder=1)
# ax.fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3),zorder=1)
# ax.fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2),zorder=1)
# ax.fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1),zorder=1)
# ax.set(xlabel='Conventional-Park AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
# handles, labels = ax.get_legend_handles_labels()
# tmp = ax.legend(handles[9:],labels[9:],loc='lower right',title='AEP % Diff.')
# ax.add_artist(tmp)
# tmp = ax.legend(handles[0:9],labels[0:9],loc='upper left',title='Wind Rose')
# tmp.get_frame().set_facecolor('gainsboro')
# fig.tight_layout()
# if save:
#     plt.savefig('./figures/aep_comparison.png', dpi=dpi)

# Cost Comparison
fig, ax = plt.subplots(1,1)
im = ax.scatter(time_park/1e-3,time_flowers/1e-3,c=nt,alpha=0.7,cmap='plasma',zorder=5,vmin=2,vmax=500)
# im = ax.scatter(time_park/1e-3,time_flowers/1e-3,c=nt,cmap='plasma',zorder=5,vmin=2,vmax=500)
ax.set(xscale='log',yscale='log',aspect='equal')
plt.autoscale(False)
plt.colorbar(im,ax=ax,label='Number of Turbines',fraction=0.046,pad=0.04,shrink=0.75)
xlim = ax.get_ylim()
cmap = cm.get_cmap('Greens')
# ylim = ax.get_xlim()
ax.fill_betweenx(xlim,[xlim[0], xlim[1]],[10*xlim[0], 10*xlim[1]],alpha=0.4,label='1x-10x Speed',color=cmap(0.2))
ax.fill_betweenx(xlim,[10*xlim[0], 10*xlim[1]],[20*xlim[0], 20*xlim[1]],alpha=0.4,label='10x-20x Speed',color=cmap(0.5))
ax.fill_betweenx(xlim,[20*xlim[0], 20*xlim[1]],[50*xlim[0], 50*xlim[1]],alpha=0.4,label='20x-50x Speed',color=cmap(0.8))
ax.set(xlabel='Conventional-Park Time [ms]',ylabel='FLOWERS Time [ms]')
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_comparison.png', dpi=dpi)

# Cost Scaling
def power_law(x, a, b):
    return a*np.power(x, b)

# time_flowers /= np.max(time_flowers)
# time_park /= np.max(time_park)

# sorting = nt.argsort()
# idx = np.where(nt[sorting] > 30)[0][0]
# idx2 = np.where(nt[sorting] > 80)[0][0]
# p , _ = curve_fit(power_law,nt,time_flowers)
# q = np.polyfit(nt[sorting[idx:idx2]],time_park[sorting[idx:idx2]],1)
# q , _ = curve_fit(power_law,nt[sorting[idx:]],time_park[sorting[idx:]] - 0.015)
# r , _ = curve_fit(power_law,nt[sorting[:idx]],time_park[sorting[:idx]])
num_turbs = np.linspace(2,500,endpoint=True)
num_turbs1 = np.linspace(30,80,endpoint=True)
num_turbs2 = np.linspace(80,500,endpoint=True)

# fig, ax = plt.subplots(1,1)
# ax.scatter(nt,time_park/1e-3, s=40,linewidth=1, alpha=0.25,facecolors='none', edgecolors='tab:orange',zorder=2)
# ax.plot(num_turbs1,1.65*num_turbs1**(1)-12, '--', color='tab:orange', linewidth=2,label='Conventional: $\mathcal{O}(N^1)$')
# ax.plot(num_turbs2,0.107*num_turbs2**(1.532)+25, '--', color='tab:orange', linewidth=2,label='Conventional: $\mathcal{O}(N^1)$')
# ax.set(xlabel='Number of Terms',ylabel='AEP Evaluation Time [ms]')

fig, ax = plt.subplots(1,1)
ax.scatter(nt,time_flowers/1e-3,s=10,linewidth=1, alpha=0.5, facecolors='tab:blue', edgecolors='tab:blue', label='FLOWERS')
ax.scatter(nt,time_park/1e-3,s=10,linewidth=1, alpha=0.5, facecolors='tab:orange', edgecolors='tab:orange', label='Conventional')
ax.plot(num_turbs, 0.8*(7.3e-4*num_turbs**(2)+0.35), '--', color='tab:blue', linewidth=3)
ax.plot(num_turbs1, 1.4*(1.65*num_turbs1**(1)-12), '--', color='tab:orange', linewidth=3)
ax.plot(num_turbs2, 0.8*(0.107*num_turbs2**(1.532)+25), '--', color='tab:orange', linewidth=3)
# ax.plot([0, 100], 1*np.array([25,1e3*np.max(time_park)]), '--', color='tab:orange', linewidth=2)
# ax.plot(range(1,101),power_law(range(1,101),p[0],p[1]),'--',linewidth=2, %(p[1]))
# ax.plot(range(31,101),power_law(range(31,101),q[0],q[1]),'--',linewidth=2,label='Conventional: $\mathcal{O}(N^{%.1f})$' %(q[1]))
# ax.plot(range(1,31),power_law(range(1,31),r[0],r[1]),'--',color="tab:orange",linewidth=2,label='')
ax.text(185,10,'$\mathcal{O}(N^2)$',color='tab:blue',fontsize=14)
ax.text(10,160,'$\mathcal{O}(N)$',color='tab:orange',fontsize=14)
ax.text(265,270,'$\mathcal{O}(N^{1.5})$',color='tab:orange',fontsize=14)
ax.set(xlabel='Number of Turbines, $N$',ylabel='AEP Evaluation Time [ms]',yscale='log')
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_turbines.png', dpi=dpi)

###########################################################################
# RANDOMIZED RESOLUTION
###########################################################################
var, aep_flowers, aep_park, time_flowers, time_park = pickle.load(open('solutions/aep1.p','rb'))
N_terms = var[0]
N_bins = var[1]

# time_flowers /= np.max(time_flowers)
# time_park /= np.max(time_park)

p = np.polyfit(time_flowers,N_terms,1)


# sorting = time_park.argsort()
# idx = np.where(time_park[sorting] > 0.4)[0][0]
# q = np.polyfit(time_park[sorting[idx:]],N_bins[sorting[idx:]],1)
# r = np.polyfit(time_park[sorting[:idx]],N_bins[sorting[:idx]],1)

n_terms = np.linspace(2,180,endpoint=True)
n_bins = np.linspace(40,360,endpoint=True)

# fig, ax = plt.subplots(1,1)
# ax.scatter(N_bins,time_park/1e-3, s=40,linewidth=1, alpha=0.25,facecolors='none', edgecolors='tab:orange',zorder=2)
# ax.plot(n_bins,0.16*n_bins**(1)+5.8e1, '--', color='tab:orange', linewidth=2,label='Conventional: $\mathcal{O}(N^1)$')
# ax.set(xlabel='Number of Terms',ylabel='AEP Evaluation Time [ms]')

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
ax.scatter(time_flowers/1e-3,N_terms,s=10,linewidth=1, alpha=0.5, facecolors='tab:blue', edgecolors='tab:blue',zorder=3,label='FLOWERS')
ax2.scatter(time_park/1e-3,N_bins,s=10,linewidth=1, alpha=0.5, facecolors='tab:orange', edgecolors='tab:orange',zorder=2)
ax.plot(0.7*(1.67e-1*n_terms**(1)+0.4),n_terms, '--', color='tab:blue', linewidth=3)
ax2.plot(0.7*(0.16*n_bins**(1)+5.8e1), n_bins, '--', color='tab:orange', linewidth=3)
# ax.plot(xrange,p[0]*xrange+p[1],'--',linewidth=2,color='tab:blue')
# ax2.plot(xrange2,q[0]*xrange2+q[1],'--',linewidth=2,color='tab:orange')
# ax2.plot(xrange1,r[0]*xrange1+r[1],'--',linewidth=2,color='tab:orange')
ax.set(xlabel='AEP Evaluation Time [ms]',xscale='log')
ax.set_ylabel('Number of Fourier Modes, $M$', color='tab:blue')
ax.tick_params(axis='y',labelcolor='tab:blue')
ax2.set_ylabel('Number of Wind Direction Bins, $\mathcal{D}$', color='tab:orange')
ax2.tick_params(axis='y',labelcolor='tab:orange')
ax.scatter([],[],s=10,linewidth=1, alpha=0.5, facecolors='tab:orange', edgecolors='tab:orange',label='Conventional')
ax2.text(3,107,'$\mathcal{O}(M)$',color='tab:blue',fontsize=14)
ax2.text(30,166,'$\mathcal{O}(\mathcal{D})$',color='tab:orange',fontsize=14)
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_resolution.png', dpi=dpi)

###########################################################################
# LAYOUT MUTATION
###########################################################################
x_all, y_all, aep_flowers, aep_park, aep_gauss, resolution, wind_rose = pickle.load(open('solutions/aep2.p','rb'))
x_all /= 126.
y_all /= 126.
N = len(x_all) - 1
flowers_terms = resolution[0]
floris_resolution = resolution[1]

fig = plt.figure(figsize=(9,5))
ax0 = plt.subplot2grid((3,2),(0,0),rowspan=3)
ax1 = plt.subplot2grid((3,2),(0,1))
ax2 = plt.subplot2grid((3,2),(1,1))
ax3 = plt.subplot2grid((3,2),(2,1))

ax0.set(aspect='equal', xlim=[-3,17], ylim=[-3,17], xticks=[0,5,10,15], yticks=[0,5,10,15], xlabel='x/D', ylabel='y/D')
ax1.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax2.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax3.set(xlim=[0,N], ylim=[0.975,1.025], xlabel='Iteration', ylabel='Normalized AEP')

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = cm.get_cmap('Reds')

patches = []
for n in range(len(x_all[0])):
    patches.append(ax0.add_patch(plt.Circle((x_all[0,n], y_all[0,n]), 1/2, color='tab:red')))

# line0, = ax0.plot([],[],"o",color='tab:red',markersize=12)
line1, = ax1.plot([],[],"-o",color=colors[0],markersize=3)
line2, = ax1.plot([],[],":o",color=colors[1],markersize=3)
line3, = ax1.plot([],[],"--o",color=colors[2],markersize=3)
line4, = ax2.plot([],[],"-o",color=colors[4],markersize=3)
line5, = ax2.plot([],[],":o",color=colors[5],markersize=3)
line6, = ax2.plot([],[],"--o",color=colors[6],markersize=3)
line7, = ax3.plot([],[],"-o",color=colors[8],markersize=3)
line8, = ax3.plot([],[],":o",color=colors[9],markersize=3)
line9, = ax3.plot([],[],"--o",color=colors[10],markersize=3)
divider = make_axes_locatable(ax0)
cbar_ax = divider.append_axes('top', size='5%', pad=0.05)
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=co.Normalize(vmin=0,vmax=N)),cax=cbar_ax,label='Iteration',orientation='horizontal')
cbar_ax.xaxis.set_ticks_position('top')
cbar_ax.xaxis.set_label_position('top')
fig.tight_layout()
ax1.legend(['FLOWERS: %.0f' %(flowers_terms[0]), 'FLOWERS: %.0f' %(flowers_terms[1]), 'FLOWERS: %.0f' %(flowers_terms[2])],loc='lower left',fontsize=8)
ax2.legend(['Park: $%.0f^\circ$' %(floris_resolution[0]),'Park: $%.0f^\circ$' %(floris_resolution[1]),'Park: $%.0f^\circ$' %(floris_resolution[2])],loc='lower left',fontsize=8)
ax3.legend(['Gauss: $%.0f^\circ$' %(floris_resolution[0]),'Gauss: $%.0f^\circ$' %(floris_resolution[1]),'Gauss: $%.0f^\circ$' %(floris_resolution[2])],loc='lower left',fontsize=8)

if save:
    plt.savefig('./figures/aep_mutation0.png', dpi=dpi)

# Function to update turbine positions
def animate1(i):
    for n in range(len(x_all[0])):
        patches[n].center = x_all[i,n], y_all[i,n]
    # line0.set_data(x_all[i], y_all[i])
    if i >= 1:
        for n in range(len(x_all[0])):
            ax0.plot([x_all[i-1,n],x_all[i,n]],[y_all[i-1,n],y_all[i,n]],"-o",color=cmap(i/N),markersize=3)
    line1.set_data(range(i+1),aep_flowers[0,0:i+1])
    line2.set_data(range(i+1),aep_flowers[1,0:i+1])
    line3.set_data(range(i+1),aep_flowers[2,0:i+1])
    line4.set_data(range(i+1),aep_park[0,0:i+1])
    line5.set_data(range(i+1),aep_park[1,0:i+1])
    line6.set_data(range(i+1),aep_park[2,0:i+1])
    line7.set_data(range(i+1),aep_gauss[0,0:i+1])
    line8.set_data(range(i+1),aep_gauss[1,0:i+1])
    line9.set_data(range(i+1),aep_gauss[2,0:i+1])
    ax1.legend(['FLOWERS: %.0f' %(flowers_terms[0]), 'FLOWERS: %.0f' %(flowers_terms[1]), 'FLOWERS: %.0f' %(flowers_terms[2])],loc='lower left',fontsize=8)
    ax2.legend(['Park: $%.0f^\circ$' %(floris_resolution[0]),'Park: $%.0f^\circ$' %(floris_resolution[1]),'Park: $%.0f^\circ$' %(floris_resolution[2])],loc='lower left',fontsize=8)
    ax3.legend(['Gauss: $%.0f^\circ$' %(floris_resolution[0]),'Gauss: $%.0f^\circ$' %(floris_resolution[1]),'Gauss: $%.0f^\circ$' %(floris_resolution[2])],loc='lower left',fontsize=8)
    return line1, line2, line3, line4, line5, line6, line7, line8, line9

# Animation
mov = animation.FuncAnimation(fig, animate1, frames=N+1, repeat=False)
if save:
    mov.save('./figures/aep_mutation.gif', dpi=dpi, bitrate=500)
    plt.savefig('./figures/aep_mutation.png', dpi=dpi)

###########################################################################
# SOLUTION SPACE SMOOTHNESS
###########################################################################
xx, yy, flowers_aep, park_aep, gauss_aep, terms, conv_resolution, wind_rose, X0, Y0, boundary = pickle.load(open('solutions/aep3.p','rb'))

boundary = np.append(boundary,np.reshape(boundary[:,0],(2,1)),axis=1)

fig3, ax5 = plt.subplots(1,3,figsize=(11,4))

for i in [0,1,2]:
    ax5[i].fill_between([-1,15],14,15,color='lightgray')
    ax5[i].fill_between([-1,15],-1,0,color='lightgray')
    ax5[i].fill_betweenx([-1,15],14,15,color='lightgray')
    ax5[i].fill_betweenx([-1,15],-1,0,color='lightgray')
    ax5[i].plot(boundary[0],boundary[1],linewidth=2,color="black",zorder=6)
    circ1=[]
    circ2=[]
    for n in range(len(X0)):
        circ1.append(plt.Circle((X0[n], Y0[n]), 1))
        circ2.append(plt.Circle((X0[n], Y0[n]), 1/2))
    coll1 = coll.PatchCollection(circ1, color='lightgray',zorder=5)
    coll2 = coll.PatchCollection(circ2, color='black', zorder=8)
    ax5[i].add_collection(coll1)
    ax5[i].add_collection(coll2)


ax5[0].contour(xx,yy,flowers_aep[-2],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax5[0].set(
    title='FLOWERS: %.0f'%(terms[-2]), 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax5[1].contour(xx,yy,park_aep[4],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax5[1].set(
    title='Conventional-Park: %.0f'%(conv_resolution[4]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax5[2].contour(xx,yy,gauss_aep[4],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax5[2].set(
    title='Conventional-Gauss: %.0f'%(conv_resolution[4]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
fig3.tight_layout()
fig3.subplots_adjust(right=0.85)
cbar_ax = fig3.add_axes([0.88, 0.205, 0.02, 0.62])
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.94,vmax=1.)),cax=cbar_ax,label='Normalized AEP')
if save:
    plt.savefig('./figures/aep_smooth.png', dpi=dpi)

fig2, ax4 = plt.subplots(1,3,figsize=(11,4))

for i in [0,1,2]:
    ax4[i].fill_between([-1,15],14,15,color='lightgray')
    ax4[i].fill_between([-1,15],-1,0,color='lightgray')
    ax4[i].fill_betweenx([-1,15],14,15,color='lightgray')
    ax4[i].fill_betweenx([-1,15],-1,0,color='lightgray')
    ax4[i].plot(boundary[0],boundary[1],linewidth=2,color="black",zorder=6)
    circ1=[]
    circ2=[]
    for n in range(len(X0)):
        circ1.append(plt.Circle((X0[n], Y0[n]), 1))
        circ2.append(plt.Circle((X0[n], Y0[n]), 1/2))
    coll1 = coll.PatchCollection(circ1, color='lightgray',zorder=5)
    coll2 = coll.PatchCollection(circ2, color='black', zorder=8)
    ax4[i].add_collection(coll1)
    ax4[i].add_collection(coll2)

ax4[0].contour(xx,yy,flowers_aep[-1],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax4[0].set(
    title='FLOWERS: %.0f'%(terms[-1]), 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax4[1].contour(xx,yy,park_aep[-1],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax4[1].set(
    title='Conventional-Park: %.0f'%(conv_resolution[-1]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax4[2].contour(xx,yy,gauss_aep[-1],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
ax4[2].set(
    title='Conventional-Gauss: %.0f'%(conv_resolution[-1]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
fig2.tight_layout()
fig2.subplots_adjust(right=0.85)
cbar_ax = fig2.add_axes([0.88, 0.205, 0.02, 0.62])
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.94,vmax=1.)),cax=cbar_ax,label='Normalized AEP')

def animate2(i):
    ax4[0].clear()
    ax4[1].clear()
    ax4[2].clear()

    for b in [0,1,2]:
        ax4[b].fill_between([-1,15],14,15,color='lightgray')
        ax4[b].fill_between([-1,15],-1,0,color='lightgray')
        ax4[b].fill_betweenx([-1,15],14,15,color='lightgray')
        ax4[b].fill_betweenx([-1,15],-1,0,color='lightgray')
        ax4[b].plot(boundary[0],boundary[1],linewidth=2,color="black",zorder=6)
        circ1=[]
        circ2=[]
        for n in range(len(X0)):
            circ1.append(plt.Circle((X0[n], Y0[n]), 1))
            circ2.append(plt.Circle((X0[n], Y0[n]), 1/2))
        coll1 = coll.PatchCollection(circ1, color='lightgray',zorder=5)
        coll2 = coll.PatchCollection(circ2, color='black', zorder=8)
        ax4[b].add_collection(coll1)
        ax4[b].add_collection(coll2)
    
    ax4[0].contour(xx,yy,flowers_aep[-1-i],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
    ax4[0].set(
        title='FLOWERS: %.0f'%(terms[-1-i]), 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax4[1].contour(xx,yy,park_aep[-1-i],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
    ax4[1].set(
        title='Conventional-Park: %.0f'%(conv_resolution[-1-i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax4[2].contour(xx,yy,gauss_aep[-1-i],levels=np.linspace(0.94,1.,50,endpoint=True),cmap='viridis',vmin=0.94,vmax=1.)
    ax4[2].set(
        title='Conventional-Gauss: %.0f'%(conv_resolution[-1-i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )

mov2 = animation.FuncAnimation(fig2, animate2, interval=1000, frames=len(conv_resolution), repeat=True)
if save:
    mov2.save('./figures/aep_smooth.gif', dpi=dpi, bitrate=500)

plt.show()