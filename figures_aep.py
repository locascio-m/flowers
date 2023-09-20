import numpy as np
import tools as tl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as co
import pickle
import visualization as vis
from scipy.optimize import curve_fit
import matplotlib.animation as animation

save = False

###########################################################################
# RANDOMIZED LAYOUT / WIND ROSE
###########################################################################
var, aep_flowers, aep_park, time_flowers, time_park = pickle.load(open('solutions/aep0.p','rb'))
wr = var[0]
nt = var[1]

aep_flowers /= 1e9
aep_park /= 1e9

# AEP Comparison
fig, ax = plt.subplots(1,1)
markers = ['o','v','^','s','P','*','X','D','p']
cmap2 = cm.get_cmap('hot')
for i in range(9):
    p = np.ma.polyfit(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),1)
    xrange = np.array([0,np.max(np.ma.masked_where(wr!=i+1,aep_park))])
    yrange = p[0]*xrange + p[1]
    im = ax.scatter(np.ma.masked_where(wr!=i+1,aep_park),np.ma.masked_where(wr!=i+1,aep_flowers),marker=markers[i],label='WR ' + str(i+1),color=cmap2(0.1*i),zorder=15)
    ax.plot(xrange,yrange,'--',zorder=14,color=cmap2(0.1*i))
xlim = ax.get_xlim()
cmap = cm.get_cmap('coolwarm_r')
ax.plot([0,xlim[1]],[0,xlim[1]],'k',linewidth=2, label='0%',zorder=10)
ax.fill_between([0, xlim[1]],[0, 0.95*xlim[1]],[0, 0.99*xlim[1]],alpha=0.4,label='[-5%,-1%]',color=cmap(0.4),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.9*xlim[1]],[0, 0.95*xlim[1]],alpha=0.4,label='[-10%,-5%]',color=cmap(0.3),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.8*xlim[1]],[0, 0.9*xlim[1]],alpha=0.4,label='[-20%,-10%]',color=cmap(0.2),zorder=1)
ax.fill_between([0, xlim[1]],[0, 0.7*xlim[1]],[0, 0.8*xlim[1]],alpha=0.4,label='[-30%,-20%]',color=cmap(0.1),zorder=1)
ax.set(xlabel='Conventional AEP [GWh]',ylabel='FLOWERS AEP [GWh]')
handles, labels = ax.get_legend_handles_labels()
tmp = ax.legend(handles[9:],labels[9:],loc='lower right')
ax.add_artist(tmp)
ax.legend(handles[0:9],labels[0:9],loc='upper left')
fig.tight_layout()
if save:
    plt.savefig('./figures/aep_comparison.png')

# Cost Comparison
fig, ax = plt.subplots(1,1)
im = ax.scatter(time_park,time_flowers,c=nt,cmap='plasma',zorder=5)
ax.set(xscale='log',yscale='log',aspect='equal')
plt.autoscale(False)
plt.colorbar(im,ax=ax,label='Number of Turbines')
xlim = ax.get_ylim()
cmap = cm.get_cmap('Greens')
# ylim = ax.get_xlim()
ax.fill_betweenx(xlim,[xlim[0], xlim[1]],[10*xlim[0], 10*xlim[1]],alpha=0.4,label='1x-10x Speed',color=cmap(0.2))
ax.fill_betweenx(xlim,[10*xlim[0], 10*xlim[1]],[20*xlim[0], 20*xlim[1]],alpha=0.4,label='10x-20x Speed',color=cmap(0.4))
ax.fill_betweenx(xlim,[20*xlim[0], 20*xlim[1]],[50*xlim[0], 50*xlim[1]],alpha=0.4,label='20x-50x Speed',color=cmap(0.6))
ax.fill_betweenx(xlim,[50*xlim[0], 50*xlim[1]],[100*xlim[0], 100*xlim[1]],alpha=0.4,label='50x-100x Speed',color=cmap(0.999))
ax.set(xlabel='Conventional Time [s]',ylabel='FLOWERS Time [s]')
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_comparison.png')

# Cost Scaling
def power_law(x, a, b):
    return a*np.power(x, b)

p , _ = curve_fit(power_law,nt,time_flowers)
q , _ = curve_fit(power_law,nt,time_park)

fig, ax = plt.subplots(1,1)
ax.scatter(nt,time_flowers,s=40,linewidth=2, alpha=0.75, facecolors='none', edgecolors='tab:blue')
ax.scatter(nt,time_park,s=40,linewidth=2, alpha=0.75, facecolors='none', edgecolors='tab:orange')
ax.plot(range(1,101),power_law(range(1,101),p[0],p[1]),'--',linewidth=2,label='FLOWERS: $\mathcal{O}(N^{%.1f})$' %(p[1]))
ax.plot(range(1,101),power_law(range(1,101),q[0],q[1]),'--',linewidth=2,label='Conventional: $\mathcal{O}(N^{%.1f})$' %(q[1]))
ax.set(xlabel='Number of Turbines',ylabel='AEP Evaluation Time [s]')
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_turbines.png')

###########################################################################
# RANDOMIZED RESOLUTION
###########################################################################
var, aep_flowers, aep_park, time_flowers, time_park = pickle.load(open('solutions/aep1.p','rb'))
N_terms = var[0]
N_bins = var[1]

time_flowers /= np.max(time_flowers)
time_park /= np.max(time_park)

p = np.polyfit(time_flowers,N_terms,1)


sorting = time_park.argsort()
idx = np.where(time_park[sorting] > 0.4)[0][0]
q = np.polyfit(time_park[sorting[idx:]],N_bins[sorting[idx:]],1)
r = np.polyfit(time_park[sorting[:idx]],N_bins[sorting[:idx]],1)

xrange = np.array([0.,1.])
xrange1 = np.array([0.18,0.3])
xrange2 = np.array([0.5,1.])

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
ax.scatter(time_flowers,N_terms,s=40,linewidth=2, alpha=0.75,facecolors='none', edgecolors='tab:blue',label='FLOWERS: $\mathcal{O}(M)$',zorder=3)
ax2.scatter(time_park,N_bins,s=40,linewidth=2, alpha=0.75,facecolors='none', edgecolors='tab:orange',zorder=2)
ax.plot(xrange,p[0]*xrange+p[1],'--',linewidth=2,color='tab:blue')
ax2.plot(xrange2,q[0]*xrange2+q[1],'--',linewidth=2,color='tab:orange')
ax2.plot(xrange1,r[0]*xrange1+r[1],'--',linewidth=2,color='tab:orange')
ax.set(xlabel='Normalized AEP Evaluation Time [--]')
ax.set_ylabel('FLOWERS: Number of Fourier Modes', color='tab:blue')
ax.tick_params(axis='y',labelcolor='tab:blue')
ax2.set_ylabel('Conventional: Number of Wind Direction Bins', color='tab:orange')
ax2.tick_params(axis='y',labelcolor='tab:orange')
ax.scatter([],[],s=40,linewidth=2, alpha=0.75,facecolors='none', edgecolors='tab:orange',label='Conventional: $\mathcal{O}(\mathcal{D})$')
ax.legend()
fig.tight_layout()
if save:
    plt.savefig('./figures/cost_resolution.png')

###########################################################################
# LAYOUT MUTATION
###########################################################################
x_all, y_all, aep_flowers, aep_park, aep_gauss, resolution, wind_rose = pickle.load(open('solutions/aep2.p','rb'))
N = len(x_all) - 1
flowers_terms = resolution[0]
floris_resolution = resolution[1]

fig = plt.figure(figsize=(11,7))
ax0 = plt.subplot2grid((3,2),(0,0),rowspan=3)
ax1 = plt.subplot2grid((3,2),(0,1))
ax2 = plt.subplot2grid((3,2),(1,1))
ax3 = plt.subplot2grid((3,2),(2,1))

# fig, (ax0,ax1) = plt.subplots(1,2, )
ax0.set(aspect='equal', xlim=[-5,20], ylim=[-5,20], xlabel='x/D', ylabel='y/D')
ax1.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax2.set(xlim=[0,N], ylim=[0.975,1.025], ylabel='Normalized AEP',xticklabels=[])
ax3.set(xlim=[0,N], ylim=[0.975,1.025], xlabel='Iteration', ylabel='Normalized AEP')

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = cm.get_cmap('Reds')

line0, = ax0.plot([],[],"o",color='tab:red',markersize=12)
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
def animate1(i):
    line0.set_data(x_all[i]/126., y_all[i]/126.)
    line1.set_data(range(i+1),aep_flowers[0,0:i+1])
    line2.set_data(range(i+1),aep_flowers[1,0:i+1])
    line3.set_data(range(i+1),aep_flowers[2,0:i+1])
    line4.set_data(range(i+1),aep_park[0,0:i+1])
    line5.set_data(range(i+1),aep_park[1,0:i+1])
    line6.set_data(range(i+1),aep_park[2,0:i+1])
    line7.set_data(range(i+1),aep_gauss[0,0:i+1])
    line8.set_data(range(i+1),aep_gauss[1,0:i+1])
    line9.set_data(range(i+1),aep_gauss[2,0:i+1])
    ax1.legend(['FLOWERS: %.0f' %(flowers_terms[0]), 'FLOWERS: %.0f' %(flowers_terms[1]), 'FLOWERS: %.0f' %(flowers_terms[2])],loc='upper left')
    ax2.legend(['Park: $%.0f^\circ$' %(floris_resolution[0]),'Park: $%.0f^\circ$' %(floris_resolution[1]),'Park: $%.0f^\circ$' %(floris_resolution[2])],loc='upper left')
    ax3.legend(['Gauss: $%.0f^\circ$' %(floris_resolution[0]),'Gauss: $%.0f^\circ$' %(floris_resolution[1]),'Gauss: $%.0f^\circ$' %(floris_resolution[2])],loc='upper left')
    return line0, line1, line2, line3, line4, line5, line6, line7, line8, line9

# Animation
mov = animation.FuncAnimation(fig, animate1, frames=N+1, repeat=False)
if save:
    mov.save('./figures/aep_mutation.mp4')
    plt.savefig('./figures/aep_mutation.png')

###########################################################################
# SOLUTION SPACE SMOOTHNESS
###########################################################################
xx, yy, flowers_aep, park_aep, gauss_aep, terms, conv_resolution, wind_rose = pickle.load(open('solutions/aep3.p','rb'))

fig3, ax5 = plt.subplots(1,3,figsize=(11,4))

ax5[0].contour(xx,yy,flowers_aep[-2],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax5[0].set(
    title='FLOWERS: %.0f'%(terms[-2]), 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax5[1].contour(xx,yy,park_aep[4],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax5[1].set(
    title='Conventional-Park: %.0f'%(conv_resolution[4]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax5[2].contour(xx,yy,gauss_aep[4],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax5[2].set(
    title='Conventional-Gauss: %.0f'%(conv_resolution[4]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
fig3.tight_layout()
fig3.subplots_adjust(right=0.85)
cbar_ax = fig3.add_axes([0.88, 0.205, 0.02, 0.62])
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.95,vmax=1.)),cax=cbar_ax,label='Normalized AEP')
if save:
    plt.savefig('./figures/aep_smooth.png')

fig2, ax4 = plt.subplots(1,3,figsize=(11,4))

ax4[0].contour(xx,yy,flowers_aep[-1],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax4[0].set(
    title='FLOWERS: %.0f'%(terms[-1]), 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax4[1].contour(xx,yy,park_aep[-1],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax4[1].set(
    title='Conventional-Park: %.0f'%(conv_resolution[-1]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
ax4[2].contour(xx,yy,gauss_aep[-1],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
ax4[2].set(
    title='Conventional-Gauss: %.0f'%(conv_resolution[-1]) + '$^\circ$', 
    xlabel='x/D', 
    ylabel='y/D',
    aspect='equal'
)
fig2.tight_layout()
fig2.subplots_adjust(right=0.85)
cbar_ax = fig2.add_axes([0.88, 0.205, 0.02, 0.62])
cbar = plt.colorbar(cm.ScalarMappable(norm=co.Normalize(vmin=0.95,vmax=1.)),cax=cbar_ax,label='Normalized AEP')

def animate2(i):
    ax4[0].clear()
    ax4[1].clear()
    ax4[2].clear()
    
    ax4[0].contour(xx,yy,flowers_aep[-1-i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax4[0].set(
        title='FLOWERS: %.0f'%(terms[-1-i]), 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax4[1].contour(xx,yy,park_aep[-1-i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax4[1].set(
        title='Conventional-Park: %.0f'%(conv_resolution[-1-i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )
    ax4[2].contour(xx,yy,gauss_aep[-1-i],levels=np.linspace(0.95,1.,50,endpoint=True),cmap='viridis',vmin=0.95,vmax=1.)
    ax4[2].set(
        title='Conventional-Gauss: %.0f'%(conv_resolution[-1-i]) + '$^\circ$', 
        xlabel='x/D', 
        ylabel='y/D',
        aspect='equal'
    )

mov2 = animation.FuncAnimation(fig2, animate2, interval=1000, frames=len(conv_resolution), repeat=True)
if save:
    mov2.save('./figures/aep_smooth.mp4')

plt.show()