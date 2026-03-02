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
import colorsys

save = False

dpi = 600
if save == False:
    plt.rcParams['figure.dpi'] = 150
# Formatting
font = 8
plt.rc('font', size=font)          # controls default text sizes
plt.rc('axes', titlesize=font)     # fontsize of the axes title
plt.rc('axes', labelsize=font)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font)    # fontsize of the tick labels
plt.rc('legend', fontsize=font-2)    # legend fontsize
plt.rc('legend', title_fontsize=font-2)    # legend fontsize
plt.rc('figure', titlesize=font)  # fontsize of the figure title

# farm = "small"
farm = "medium"
# farm = "large"

# Define plotting parameters
color_neutral = 'tab:olive'
color_flowers = 'tab:blue'
color_conventional = 'tab:red'
color_numerical = 'tab:purple'

if farm == "small":
    time_scale = 60
    time_scale_string = '[min]'
    tperi_scale = 1
    tperi_scale_string = '[s]'
    xticks = [0,2,4,6,8,10]
    yticks = [0,2,4,6,8,10]
    timeticks = [0.,0.5,1.,1.5,2.,2.5,3.,3.5]
    iterticks = [0,100,200,300,400]
elif farm == "medium":
    time_scale = 60
    time_scale_string = '[min]'
    tperi_scale = 1
    tperi_scale_string = '[s]'
    xticks = [0,5,10,15,20,25,30,35]
    yticks = [0,5,10,15,20,25,30,35]
    timeticks = [0,20,40,60,80,100,120,140,160]
    iterticks = [0,100,200,300,400,500]
elif farm == "large":
    time_scale = 3600
    time_scale_string = '[hr]'
    tperi_scale = 60
    tperi_scale_string = '[min]'
    xticks = [0,25,50,75,100,125]
    yticks = [0,25,50,75,100,125,150]
    timeticks = [0,20,40,60,80,100]
    iterticks = [0,200,400,600,800,1000,1200,1400]

# Farm definitions (VERSION 1)
# fig = plt.figure(figsize=(6.5,6))
# axLG = plt.subplot2grid((3,3),(0,0),colspan=2,rowspan=3)
# axSMwr = plt.subplot2grid((3,3),(0,2),projection='polar')
# axMDwr = plt.subplot2grid((3,3),(1,2),projection='polar')
# axLGwr = plt.subplot2grid((3,3),(2,2),projection='polar')

# xshift_SM = 75
# yshift_SM = 250
# xshift_MD = 120
# yshift_MD = 175

# # Large farm
# wr = tl.load_wind_rose(6)
# xx, yy, boundaries = tl.load_layout(41, "large")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i], yy[i]), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# axLG.add_collection(layouts)

# axLG.plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
# xlim_full = axLG.get_xlim()
# ylim_full = axLG.get_ylim()
# axLG.set(
#     xlabel='$x/D$', 
#     ylabel='$y/D$', 
#     aspect='equal',
#     xlim=[xlim_full[0],300], 
#     ylim=[ylim_full[0],300], 
#     xticks=np.arange(0,325,25),
#     yticks=np.arange(0,325,25),
#     xticklabels=['0','25','50','75','100','125','','','','','','',''],
#     yticklabels=['0','25','50','75','100','125','150','','','','','','']
# )
# axLG.grid(linestyle=':',linewidth=0.5)
# axLG.set_axisbelow(True)

# vis.plot_wind_rose(wr, ax=axLGwr)
# h, l = axLGwr.get_legend_handles_labels()
# h = h[:5]
# l = l[:5]
# axLGwr.get_legend().remove()

# # Medium farm
# wr = tl.load_wind_rose(1)
# xx, yy, boundaries = tl.load_layout(20, "medium")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i]+xshift_MD, yy[i]+yshift_MD), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# layouts_full = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# axLG.add_collection(layouts_full)

# axLG.plot(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='k',linewidth=1.5,zorder=1)

# axMD = zoomed_inset_axes(axLG, zoom=3, bbox_to_anchor=[0.95,0.5], bbox_transform=axLG.transAxes)
# axMD.add_collection(layouts)
# axMD.plot(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='k',linewidth=1.5,zorder=1)
# axMD.set(
#     aspect='equal',
#     xticks=np.array([0,5,10,15,20,25,30,35])+xshift_MD,
#     yticks=np.array([0,5,10,15,20,25,30,35])+yshift_MD,
#     xticklabels=[0,5,10,15,20,25,30,35],
#     yticklabels=[0,5,10,15,20,25,30,35],
# )
# axMD.grid(linestyle=':',linewidth=0.5)
# axMD.set_axisbelow(True)
# # axMD.margins(0.15)
# mark_inset(axLG,axMD,loc1=1,loc2=3, facecolor="none",edgecolor='lightgrey',zorder=0.5)

# vis.plot_wind_rose(wr, ax=axMDwr)
# axMDwr.get_legend().remove()

# # Small farm
# wr = tl.load_wind_rose(8)
# xx, yy, boundaries = tl.load_layout(10, "small")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i]+xshift_SM, yy[i]+yshift_SM), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# layouts_full = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# axLG.add_collection(layouts_full)

# axLG.plot(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='k',linewidth=1.5,zorder=1)

# axSM = zoomed_inset_axes(axLG, zoom=8, bbox_to_anchor=[0.98,0.98], bbox_transform=axLG.transAxes)
# axSM.add_collection(layouts)
# axSM.plot(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='k',linewidth=1.5,zorder=1)
# axSM.set(
#     aspect='equal',
#     xticks=np.array([0,2,4,6,8,10])+xshift_SM,
#     yticks=np.array([0,2,4,6,8,10])+yshift_SM,
#     xticklabels=[0,2,4,6,8,10],
#     yticklabels=[0,2,4,6,8,10],
# )
# axSM.grid(linestyle=':',linewidth=0.5)
# axSM.set_axisbelow(True)
# # axSM.margins(0.2)
# mark_inset(axLG,axSM,loc1=2,loc2=3,facecolor="none",edgecolor='lightgrey',zorder=0.5)

# vis.plot_wind_rose(wr, ax=axSMwr)
# axSMwr.get_legend().remove()

# axLG.legend(reversed(h), l, ncol=5,loc='lower center',bbox_to_anchor=(0.5,-0.25))
# axLG.text(12.5,135,'(LG)',fontweight='bold', horizontalalignment='center', verticalalignment='center')
# axLG.text(212.5,160,'(MD)',fontweight='bold', horizontalalignment='center', verticalalignment='center')
# axLG.text(162.5,260,'(SM)',fontweight='bold', horizontalalignment='center', verticalalignment='center')
# fig.tight_layout()
# axLGwr.text(-0.15,0.8,'(LG)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axLGwr.transAxes)
# axMDwr.text(-0.15,0.8,'(MD)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axMDwr.transAxes)
# axSMwr.text(-0.15,0.8,'(SM)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axSMwr.transAxes)

# END VERSION 1

# # Farm definitions (VERSION 2)
# fig = plt.figure(figsize=(6.5,5.5))
# ax = plt.subplot2grid((3,3),(0,0),colspan=2,rowspan=3)
# axSMwr = plt.subplot2grid((3,3),(2,2),projection='polar')
# axMDwr = plt.subplot2grid((3,3),(1,2),projection='polar')
# axLGwr = plt.subplot2grid((3,3),(0,2),projection='polar')

# xshift_SM = 62.5
# yshift_SM = 62.5
# xshift_MD = 50
# yshift_MD = 50

# # Large farm
# wr = tl.load_wind_rose(6)
# xx, yy, boundaries = tl.load_layout(41, "large")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# # ax.add_patch(plt.Circle((25,150),1/2,color='k'))
# # ax.scatter(25,150,5,marker='1',color='r',linewidth=1)
# # ax.scatter([],[],8,color=color_neutral,label='LG: $N = 250$')
# # ax.scatter([],[],8,color=color_neutral,label='MD: $N = 50$')
# # ax.scatter([],[],8,color=color_neutral,label='SM: $N = 10$')
# # tmp = ax.legend(loc='upper left')
# # ax.add_artist(tmp)

# # layout = []
# # for i in range(len(xx)):
# #     layout.append(plt.Circle((xx[i], yy[i]), 1/2))
# # layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# # ax.add_collection(layouts)

# ax.scatter(xx,yy,3,color=color_neutral,marker='o',label='LG: $N = 250$',zorder=2)
# ax.plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
# ax.set(
#     xlabel='$x/D$', 
#     ylabel='$y/D$', 
#     aspect='equal',
#     xticks=[0,25,50,75,100,125],
#     yticks=[0,25,50,75,100,125,150],
# )

# vis.plot_wind_rose(wr, ax=axLGwr)
# h, l = axLGwr.get_legend_handles_labels()
# h = h[:5]
# l = l[:5]
# axLGwr.get_legend().remove()

# # Medium farm
# wr = tl.load_wind_rose(1)
# xx, yy, boundaries = tl.load_layout(20, "medium")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# # layout = []
# # for i in range(len(xx)):
# #     layout.append(plt.Circle((xx[i]+xshift_MD, yy[i]+yshift_MD), 1/2))
# # layouts = coll.PatchCollection(layout, color=color_neutral,hatch='/',zorder=5)
# # ax.add_collection(layouts)

# ax.scatter(xx+xshift_MD,yy+yshift_MD,5,color=color_neutral,marker='^',linewidth=0.5,label='MD: $N = 50$',zorder=5)
# ax.plot(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='k',linewidth=1.5,zorder=4)
# ax.fill(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='white',zorder=3)

# vis.plot_wind_rose(wr, ax=axMDwr)
# axMDwr.get_legend().remove()

# # Small farm
# wr = tl.load_wind_rose(8)
# xx, yy, boundaries = tl.load_layout(10, "small")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# # layout = []
# # for i in range(len(xx)):
# #     layout.append(plt.Circle((xx[i]+xshift_SM, yy[i]+yshift_SM), 1/2))
# # layouts = coll.PatchCollection(layout,color=color_neutral, zorder=8)
# # ax.add_collection(layouts)

# ax.scatter(xx+xshift_SM,yy+yshift_SM,5,color=color_neutral,marker='v',linewidth=0.5,label='SM: $N = 10$',zorder=8)
# ax.plot(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='k',linewidth=1.5,zorder=7)
# ax.fill(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='white',zorder=6)

# vis.plot_wind_rose(wr, ax=axSMwr)
# axSMwr.get_legend().remove()

# tmp = ax.legend(loc='upper left',prop={'weight':'bold'})
# ax.add_artist(tmp)
# ax.margins(0.08)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# ax.set(xlim=xlim,ylim=ylim)
# ax.hlines([0,25,50,75,100,125,150],xlim[0],xlim[1],colors='darkgrey',linewidths=0.5,linestyles=":",zorder=10)
# ax.vlines(0,ylim[0],ylim[1],colors='darkgrey',linewidths=0.5,linestyles=":",zorder=1)
# ax.vlines([25,50,75,100,125],ylim[0],ylim[1],colors='darkgrey',linewidths=0.5,linestyles=":",zorder=11)
# ax.legend(reversed(h), l, ncol=5,loc='lower center',bbox_to_anchor=(0.5,-0.15))
# ax.text(62.5,165,'(LG)',fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=12)
# ax.text(62.5,92.5,'(MD)',fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=13)
# ax.text(67.5,77.5,'(SM)',fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=14)
# ax.set_axisbelow(False)
# fig.tight_layout()
# axLGwr.text(-0.2,0.8,'(LG)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axLGwr.transAxes)
# axMDwr.text(-0.2,0.8,'(MD)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axMDwr.transAxes)
# axSMwr.text(-0.2,0.8,'(SM)',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=axSMwr.transAxes)

# END VERSION 2

# # Farm definitions (VERSION 3)
# fig = plt.figure(figsize=(6.5,5.25))
# ax0 = fig.add_subplot(2,3,1)
# ax1 = fig.add_subplot(2,3,2)
# ax2 = fig.add_subplot(2,3,3)
# ax3 = fig.add_subplot(2,3,4, projection='polar')
# ax4 = fig.add_subplot(2,3,5, projection='polar')
# ax5 = fig.add_subplot(2,3,6, projection='polar')
# ax = [[ax0,ax1,ax2],[ax3,ax4,ax5]]

# # Small farm
# wr = tl.load_wind_rose(8)
# xx, yy, boundaries = tl.load_layout(10, "small")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i], yy[i]), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# ax[0][0].add_collection(layouts)

# ax[0][0].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
# ax[0][0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 10$', xticks=[0,2,4,6,8,10],yticks=[0,2,4,6,8,10])
# ax[0][0].grid(linestyle=':',linewidth=0.5)
# ax[0][0].set_axisbelow(True)

# vis.plot_wind_rose(wr, ax=ax[1][0])
# ax[1][0].get_legend().remove()

# # Medium farm
# wr = tl.load_wind_rose(1)
# xx, yy, boundaries = tl.load_layout(20, "medium")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.
# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i], yy[i]), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# ax[0][1].add_collection(layouts)

# ax[0][1].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
# ax[0][1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 50$', xticks=[0,5,10,15,20,25,30,35],yticks=[0,5,10,15,20,25,30,35])
# ax[0][1].grid(linestyle=':',linewidth=0.5)
# ax[0][1].set_axisbelow(True)

# vis.plot_wind_rose(wr, ax=ax[1][1])
# ax[1][1].get_legend().remove()

# # Large farm
# wr = tl.load_wind_rose(6)
# xx, yy, boundaries = tl.load_layout(41, "large")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.
# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i], yy[i]), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
# ax[0][2].add_collection(layouts)

# ax[0][2].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
# ax[0][2].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 250$', xticks=[0,25,50,75,100,125],yticks=[0,25,50,75,100,125,150])
# # ax[0][2].set_box_aspect(1)
# ax[0][2].grid(linestyle=':',linewidth=0.5)
# ax[0][2].set_axisbelow(True)

# vis.plot_wind_rose(wr, ax=ax[1][2])
# h, l = ax[1][2].get_legend_handles_labels()
# h = h[:5]
# l = l[:5]
# ax[1][2].get_legend().remove()

# fig.tight_layout()
# fig.subplots_adjust(top=0.9,bottom=0.08)
# fig.legend(reversed(h), l, ncol=5,loc='lower center')
# fig.text(0.199, 0.96, 'Small (SM) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')
# fig.text(0.519, 0.96, 'Medium (MD) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')
# fig.text(0.834, 0.96, 'Large (LG) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')

# END VERSION 3

# # Farm definitions (VERSION 4)
# fig = plt.figure(figsize=(6.5,6))
# ax = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=2)
# axSMwr = plt.subplot2grid((3,3),(2,0),projection='polar')
# axMDwr = plt.subplot2grid((3,3),(2,1),projection='polar')
# axLGwr = plt.subplot2grid((3,3),(2,2),projection='polar')

# xshift_SM = 390
# yshift_SM = 60
# xshift_MD = 485
# yshift_MD = 50
# xshift_LG = 600
# yshift_LG = 10

# # Large farm
# wr = tl.load_wind_rose(6)
# xx, yy, boundaries = tl.load_layout(41, "large")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i]+xshift_LG, yy[i]+yshift_LG), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)

# ax.plot(boundaries[0]+xshift_LG,boundaries[1]+yshift_LG,color='k',linewidth=1,zorder=1)

# axLG = ax.inset_axes([0.64,0.4,0.45,0.55])
# axLG.add_collection(layouts)
# axLG.plot(boundaries[0]+xshift_LG,boundaries[1]+yshift_LG,color='k',linewidth=1.5,zorder=1)
# axLG.set(
#     xlabel='$x/D$',
#     ylabel='$y/D$',
#     title='$N=250$',
#     aspect='equal',
#     xticks=np.array([0,25,50,75,100,125])+xshift_LG,
#     yticks=np.array([0,25,50,75,100,125,150])+yshift_LG,
#     xticklabels=[0,25,50,75,100,125],
#     yticklabels=[0,25,50,75,100,125,150],
# )
# axLG.grid(linestyle=':',linewidth=0.5)
# axLG.set_axisbelow(True)
# mark_inset(ax,axLG,loc1=2,loc2=4, facecolor="none",edgecolor='lightgrey',zorder=0.5)

# ax.set(
#     aspect='equal',
#     xlim=[0,1000], 
#     ylim=[0,500], 
#     xticks=[],
#     yticks=[],
# )
# ax.set_axis_off()

# vis.plot_wind_rose(wr, ax=axLGwr)
# h, l = axLGwr.get_legend_handles_labels()
# h = h[:5]
# l = l[:5]
# axLGwr.get_legend().remove()

# # Medium farm
# wr = tl.load_wind_rose(1)
# xx, yy, boundaries = tl.load_layout(20, "medium")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i]+xshift_MD, yy[i]+yshift_MD), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)

# ax.plot(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='k',linewidth=1,zorder=1)

# axMD = ax.inset_axes([0.28,0.5,0.45,0.45])
# axMD.add_collection(layouts)
# axMD.plot(boundaries[0]+xshift_MD,boundaries[1]+yshift_MD,color='k',linewidth=1.5,zorder=1)
# axMD.set(
#     xlabel='$x/D$',
#     ylabel='$y/D$',
#     title='$N=50$',
#     aspect='equal',
#     xticks=np.array([0,5,10,15,20,25,30,35])+xshift_MD,
#     yticks=np.array([0,5,10,15,20,25,30,35])+yshift_MD,
#     xticklabels=[0,5,10,15,20,25,30,35],
#     yticklabels=[0,5,10,15,20,25,30,35],
# )
# axMD.grid(linestyle=':',linewidth=0.5)
# axMD.set_axisbelow(True)
# mark_inset(ax,axMD,loc1=4,loc2=3, facecolor="none",edgecolor='lightgrey',zorder=0.5)

# vis.plot_wind_rose(wr, ax=axMDwr)
# axMDwr.get_legend().remove()

# # Small farm
# wr = tl.load_wind_rose(8)
# xx, yy, boundaries = tl.load_layout(10, "small")
# xx /= 126.
# yy /= 126.
# boundaries = np.array(boundaries).T
# boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

# layout = []
# for i in range(len(xx)):
#     layout.append(plt.Circle((xx[i]+xshift_SM, yy[i]+yshift_SM), 1/2))
# layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)

# ax.plot(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='k',linewidth=1,zorder=1)

# axSM = ax.inset_axes([-0.09,0.5,0.45,0.45])
# axSM.add_collection(layouts)
# axSM.plot(boundaries[0]+xshift_SM,boundaries[1]+yshift_SM,color='k',linewidth=1.5,zorder=1)
# axSM.set(
#     xlabel='$x/D$',
#     ylabel='$y/D$',
#     title='$N=10$',
#     aspect='equal',
#     xticks=np.array([0,2,4,6,8,10])+xshift_SM,
#     yticks=np.array([0,2,4,6,8,10])+yshift_SM,
#     xticklabels=[0,2,4,6,8,10],
#     yticklabels=[0,2,4,6,8,10],
# )
# axSM.grid(linestyle=':',linewidth=0.5)
# axSM.set_axisbelow(True)
# mark_inset(ax,axSM,loc1=1,loc2=3, facecolor="none",edgecolor='lightgrey',zorder=0.5)

# vis.plot_wind_rose(wr, ax=axSMwr)
# axSMwr.get_legend().remove()

# fig.tight_layout(w_pad=3,h_pad=0.1)
# fig.legend(reversed(h), l, ncol=5,loc='lower center',bbox_to_anchor=(0.5,0.01))
# ax.text(0.865,1.06,'Large (LG) Case',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
# ax.text(0.505,1.06,'Medium (MD) Case',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)
# ax.text(0.135,1.06,'Small (SM) Case',fontweight='bold', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)

# box = axSMwr.get_position()
# box.y0 = box.y0 + 0.05
# box.y1 = box.y1 + 0.05
# axSMwr.set_position(box)
# box = axMDwr.get_position()
# box.y0 = box.y0 + 0.05
# box.y1 = box.y1 + 0.05
# axMDwr.set_position(box)
# box = axLGwr.get_position()
# box.y0 = box.y0 + 0.05
# box.y1 = box.y1 + 0.05
# axLGwr.set_position(box)

# # END VERSION 4

# Farm definitions (VERSION 5)
fig = plt.figure(figsize=(6.5,5.25))
ax2 = fig.add_subplot(2,3,3)
ax0 = fig.add_subplot(2,3,1,sharex=ax2,sharey=ax2)
ax1 = fig.add_subplot(2,3,2,sharex=ax2,sharey=ax2)
ax3 = fig.add_subplot(2,3,4, projection='polar')
ax4 = fig.add_subplot(2,3,5, projection='polar')
ax5 = fig.add_subplot(2,3,6, projection='polar')
ax = [[ax0,ax1,ax2],[ax3,ax4,ax5]]

# Small farm
wr = tl.load_wind_rose(8)
xx, yy, boundaries = tl.load_layout(10, "small")
xx /= 126.
yy /= 126.
boundaries = np.array(boundaries).T
boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.

layout = []
for i in range(len(xx)):
    layout.append(plt.Circle((xx[i], yy[i]), 1/2))
layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
layouts_inset = coll.PatchCollection(layout, color=color_neutral,zorder=2)
ax[0][0].add_collection(layouts)

ax[0][0].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
ax[0][0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 10$')
ax[0][0].grid(linestyle=':',linewidth=0.5)
ax[0][0].set_axisbelow(True)

axSM = ax[0][0].inset_axes([0.2,0.3,0.75,0.75])
axSM.add_collection(layouts_inset)
axSM.plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
axSM.set_aspect('equal')
axSM.grid(linestyle=':',linewidth=0.5)
axSM.set_axisbelow(True)
axSM.set_xticks([0,2,4,6,8,10],[0,2,4,6,8,10],fontsize=font-2)
axSM.set_yticks([0,2,4,6,8,10],[0,2,4,6,8,10],fontsize=font-2)
mark_inset(ax[0][0],axSM,loc1=2,loc2=4, facecolor="none",edgecolor='lightgrey',zorder=0.5)

vis.plot_wind_rose(wr, ax=ax[1][0])
ax[1][0].get_legend().remove()

# Medium farm
wr = tl.load_wind_rose(1)
xx, yy, boundaries = tl.load_layout(20, "medium")
xx /= 126.
yy /= 126.
boundaries = np.array(boundaries).T
boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.
layout = []
for i in range(len(xx)):
    layout.append(plt.Circle((xx[i], yy[i]), 1/2))
layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
layouts_inset = coll.PatchCollection(layout, color=color_neutral,zorder=2)
ax[0][1].add_collection(layouts)

ax[0][1].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
ax[0][1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 50$')
ax[0][1].grid(linestyle=':',linewidth=0.5)
ax[0][1].set_axisbelow(True)

axMD = ax[0][1].inset_axes([0.2,0.3,0.75,0.75])
axMD.add_collection(layouts_inset)
axMD.plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
axMD.set_aspect('equal')
axMD.grid(linestyle=':',linewidth=0.5)
axMD.set_axisbelow(True)
axMD.set_xticks([0,5,10,15,20,25,30,35],[0,5,10,15,20,25,30,35],fontsize=font-2)
axMD.set_yticks([0,5,10,15,20,25,30,35],[0,5,10,15,20,25,30,35],fontsize=font-2)
mark_inset(ax[0][1],axMD,loc1=2,loc2=4, facecolor="none",edgecolor='lightgrey',zorder=0.5)

vis.plot_wind_rose(wr, ax=ax[1][1])
ax[1][1].get_legend().remove()

# Large farm
wr = tl.load_wind_rose(6)
xx, yy, boundaries = tl.load_layout(41, "large")
xx /= 126.
yy /= 126.
boundaries = np.array(boundaries).T
boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.
layout = []
for i in range(len(xx)):
    layout.append(plt.Circle((xx[i], yy[i]), 1/2))
layouts = coll.PatchCollection(layout, color=color_neutral,zorder=2)
ax[0][2].add_collection(layouts)

ax[0][2].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
ax[0][2].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N = 250$', xticks=[0,25,50,75,100,125],yticks=[0,25,50,75,100,125,150])
# ax[0][2].set_box_aspect(1)
ax[0][2].grid(linestyle=':',linewidth=0.5)
ax[0][2].set_axisbelow(True)

vis.plot_wind_rose(wr, ax=ax[1][2])
h, l = ax[1][2].get_legend_handles_labels()
h = h[:5]
l = l[:5]
ax[1][2].get_legend().remove()

fig.tight_layout()
fig.subplots_adjust(top=0.9,bottom=0.08)
fig.legend(reversed(h), l, ncol=5,loc='lower center')
fig.text(0.199, 0.96, 'Small (SM) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')
fig.text(0.519, 0.96, 'Medium (MD) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')
fig.text(0.834, 0.96, 'Large (LG) Case', fontweight='bold', horizontalalignment='center', verticalalignment='center')

# END VERSION 5

if save and farm == 'small':
    plt.savefig('./figures/opt_multistart_setup.png', dpi=dpi)

###########################################################################
# MULTISTART
###########################################################################

# Load files
N = 100

flowers_ad = []
flowers_fd = []
conventional_fd = []

file_base = 'solutions/opt_' + farm
fig_name = './figures/opt_' + farm + '_'

for n in range(N):
    file_name = file_base + '_flowers_analytical_' + str(n) + '.p'
    solution, wr, boundaries = pickle.load(open(file_name,'rb'))
    flowers_ad.append(solution)

    file_name = file_base + '_flowers_numerical_' + str(n) + '.p'
    solution, wr, boundaries = pickle.load(open(file_name,'rb'))
    flowers_fd.append(solution)

    file_name = file_base + '_conventional_numerical_' + str(n) + '.p'
    solution, wr, boundaries = pickle.load(open(file_name,'rb'))
    conventional_fd.append(solution)

boundaries /= 126.
boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

# Collect statistics
initial_aep = np.zeros(N)
flowers_ad_optimal_aep = np.zeros(N)
flowers_ad_time = np.zeros(N)
flowers_ad_iterations = np.zeros(N)
flowers_ad_codes = {}
flowers_fd_optimal_aep = np.zeros(N)
flowers_fd_time = np.zeros(N)
flowers_fd_iterations = np.zeros(N)
flowers_fd_codes = {}
conventional_fd_optimal_aep = np.zeros(N)
conventional_fd_time = np.zeros(N)
conventional_fd_iterations = np.zeros(N)
conventional_fd_codes = {}

for n in range(N):
    initial_aep[n] = flowers_ad[n]['init_aep']

    flowers_ad_optimal_aep[n] = flowers_ad[n]['opt_aep']
    flowers_ad_time[n] = flowers_ad[n]['total_time']
    flowers_ad_iterations[n] = flowers_ad[n]['iter']

    flowers_fd_optimal_aep[n] = flowers_fd[n]['opt_aep']
    flowers_fd_time[n] = flowers_fd[n]['total_time']
    flowers_fd_iterations[n] = flowers_fd[n]['iter']

    conventional_fd_optimal_aep[n] = conventional_fd[n]['opt_aep']
    conventional_fd_time[n] = conventional_fd[n]['total_time']
    conventional_fd_iterations[n] = conventional_fd[n]['iter']

    exit_code = flowers_ad[n]['exit_code']
    if exit_code in flowers_ad_codes:
        flowers_ad_codes[exit_code] += 1
    else:
        flowers_ad_codes[exit_code] = 1
    
    exit_code = flowers_fd[n]['exit_code']
    if exit_code in flowers_fd_codes:
        flowers_fd_codes[exit_code] += 1
    else:
        flowers_fd_codes[exit_code] = 1
    
    exit_code = conventional_fd[n]['exit_code']
    if exit_code in conventional_fd_codes:
        conventional_fd_codes[exit_code] += 1
    else:
        conventional_fd_codes[exit_code] = 1

# Calculate time per iteration
flowers_ad_tperi = flowers_ad_time / flowers_ad_iterations
flowers_fd_tperi = flowers_fd_time / flowers_fd_iterations
conventional_fd_tperi = conventional_fd_time / conventional_fd_iterations

# Calculate outliers (TURN ON/OFF)
# for n in range(N):
#     solution = flowers_ad[n]
#     xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
#     yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
#     dd = np.sqrt(xx**2 + yy**2)
#     dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
#     if np.min(dd) < 1.:
#         print("===")
#         print("FLOWERS-AD Infeasible Solution: {:d}" % n)
#         print("===")
    
#     solution = flowers_fd[n]
#     xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
#     yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
#     dd = np.sqrt(xx**2 + yy**2)
#     dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
#     if np.min(dd) < 1.:
#         print("===")
#         print("FLOWERS-FD Infeasible Solution: {:d}" % n)
#         print("===")

#     solution = conventional_fd[n]
#     xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
#     yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
#     dd = np.sqrt(xx**2 + yy**2)
#     dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
#     if np.min(dd) < 1.:
#         print("===")
#         print("Conventional-FD Infeasible Solution: %d" % n)
#         print("===")

# Define outliers
flowers_ad_outliers = np.zeros(N)
flowers_fd_outliers = np.zeros(N)
conventional_fd_outliers = np.zeros(N)
if farm == 'small':
    conventional_fd_outliers[[57,60]] = 1
elif farm == 'medium':
    conventional_fd_outliers[[13]] = 1
elif farm == 'large': # 18 is a copy of 16
    conventional_fd_outliers[[18,77]] = 1

# Mask outliers
flowers_ad_optimal_aep = np.ma.masked_where(flowers_ad_outliers,flowers_ad_optimal_aep)
flowers_fd_optimal_aep = np.ma.masked_where(flowers_fd_outliers,flowers_fd_optimal_aep)
conventional_fd_optimal_aep = np.ma.masked_where(conventional_fd_outliers,conventional_fd_optimal_aep)

# Organize exit codes
flowers_ad_codes_reduced = {'finished successfully': 0, 'terminated after numerical difficulties': 0}
flowers_fd_codes_reduced = {'finished successfully': 0, 'terminated after numerical difficulties': 0}
conventional_fd_codes_reduced = {'finished successfully': 0, 'terminated after numerical difficulties': 0}

if farm == "large":
    flowers_ad_codes_reduced['user requested termination'] = 0
    flowers_fd_codes_reduced['user requested termination'] = 0
    conventional_fd_codes_reduced['user requested termination'] = 0

for code in flowers_ad_codes.keys(): # Small '45': convergence hangs, 398 iterations; Large '28' and '59': null space issue
    if code == 'optimality conditions satisfied' or code == 'requested accuracy could not be achieved' or code == 'ill-conditioned null-space basis':
        flowers_ad_codes_reduced['finished successfully'] += flowers_ad_codes[code]
    elif code == 'current point cannot be improved' or code == 'cannot satisfy the general constraints':
        flowers_ad_codes_reduced['terminated after numerical difficulties'] += flowers_ad_codes[code]
    elif code == 'terminated during function evaluation':
        flowers_ad_codes_reduced['user requested termination'] += flowers_ad_codes[code]

for code in flowers_fd_codes.keys():
    if code == 'optimality conditions satisfied' or code == 'requested accuracy could not be achieved':
        flowers_fd_codes_reduced['finished successfully'] += flowers_fd_codes[code]
    elif code == 'current point cannot be improved' or code == 'ill-conditioned null-space basis' or code == 'cannot satisfy the general constraints':
        flowers_fd_codes_reduced['terminated after numerical difficulties'] += flowers_fd_codes[code]
    elif code == 'terminated during function evaluation':
        flowers_fd_codes_reduced['user requested termination'] += flowers_fd_codes[code]

for code in conventional_fd_codes.keys(): 
    if code == 'optimality conditions satisfied' or code == 'requested accuracy could not be achieved':
        conventional_fd_codes_reduced['finished successfully'] += conventional_fd_codes[code]
    elif code == 'current point cannot be improved' or code == 'ill-conditioned null-space basis' or code == 'cannot satisfy the general constraints':
        conventional_fd_codes_reduced['terminated after numerical difficulties'] += conventional_fd_codes[code]
    elif code == 'terminated during function evaluation':
        conventional_fd_codes_reduced['user requested termination'] += conventional_fd_codes[code]

print("=================================================================")
print("FLOWERS-AD Median Cost: {:.1f} s".format(np.median(flowers_ad_time)))
print("FLOWERS-FD Median Cost: {:.1f} s".format(np.median(flowers_fd_time)))
print("Gradient Speed-Up: {:.2f}x".format(np.median(flowers_fd_time)/np.median(flowers_ad_time)))
print("Conventional-FD Median Cost: {:.1f} ".format(np.median(conventional_fd_time)/time_scale) + time_scale_string[1:-1])
print("Model Speed-Up: {:.2f}x".format(np.median(conventional_fd_time)/np.median(flowers_fd_time)))
print("Total Speed-Up: {:.2f}x".format(np.median(conventional_fd_time)/np.median(flowers_ad_time)))
print()
print("FLOWERS-AD Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(flowers_ad_time)/3600))
print("FLOWERS-FD Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(flowers_fd_time)/3600))
print("Gradient Speed-Up: {:.2f}x".format(np.sum(flowers_fd_time)/np.sum(flowers_ad_time)))
print("Conventional-FD Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(conventional_fd_time)/3600))
print("Model Speed-Up: {:.2f}x".format(np.sum(conventional_fd_time)/np.sum(flowers_fd_time)))
print("Total Speed-Up: {:.2f}x".format(np.sum(conventional_fd_time)/np.sum(flowers_ad_time)))
print()
print("FLOWERS-AD Normalized Std Cost: {:.1f}%".format(np.std(flowers_ad_time)/np.mean(flowers_ad_time)*100))
print("FLOWERS-FD Normalized Std Cost: {:.1f}%".format(np.std(flowers_fd_time)/np.mean(flowers_fd_time)*100))
print("Gradient Improvement: {:.2f}%".format((np.std(flowers_ad_time)/np.mean(flowers_ad_time) - np.std(flowers_fd_time)/np.mean(flowers_fd_time))/(np.std(flowers_fd_time)/np.mean(flowers_fd_time))*100))
print("Conventional-FD Normalized Std Cost: {:.1f}%".format(np.std(conventional_fd_time)/np.mean(conventional_fd_time)*100))
print("Model Improvement: {:.2f}%".format((np.std(flowers_fd_time)/np.mean(flowers_fd_time) - np.std(conventional_fd_time)/np.mean(conventional_fd_time))/(np.std(conventional_fd_time)/np.mean(conventional_fd_time))*100))
print("Total Improvement: {:.2f}%".format((np.std(flowers_ad_time)/np.mean(flowers_ad_time) - np.std(conventional_fd_time)/np.mean(conventional_fd_time))/(np.std(conventional_fd_time)/np.mean(conventional_fd_time))*100))
print()
print("FLOWERS-AD Median AEP: {:.1f} GWh".format(np.median(flowers_ad_optimal_aep)/1e9))
print("FLOWERS-FD Median AEP: {:.1f} GWh".format(np.median(flowers_fd_optimal_aep)/1e9))
print("Gradient Improvement: {:.2f}%".format((np.median(flowers_ad_optimal_aep) - np.median(flowers_fd_optimal_aep))/np.median(flowers_fd_optimal_aep)*100))
print("Conventional-FD Median AEP: {:.1f} GWh".format(np.median(conventional_fd_optimal_aep)/1e9))
print("Model Improvement: {:.2f}%".format((np.median(flowers_fd_optimal_aep) - np.median(conventional_fd_optimal_aep))/np.median(conventional_fd_optimal_aep)*100))
print("Total Improvement: {:.2f}%".format((np.median(flowers_ad_optimal_aep) - np.median(conventional_fd_optimal_aep))/np.median(conventional_fd_optimal_aep)*100))
print()
print("FLOWERS-AD Best AEP: {:.1f} GWh".format(np.max(flowers_ad_optimal_aep)/1e9))
print("FLOWERS-FD Best AEP: {:.1f} GWh".format(np.max(flowers_fd_optimal_aep)/1e9))
print("Gradient Improvement: {:.2f}%".format((np.max(flowers_ad_optimal_aep) - np.max(flowers_fd_optimal_aep))/np.max(flowers_fd_optimal_aep)*100))
print("Conventional-FD Best AEP: {:.1f} GWh".format(np.max(conventional_fd_optimal_aep)/1e9))
print("Model Improvement: {:.2f}%".format((np.max(flowers_fd_optimal_aep) - np.max(conventional_fd_optimal_aep))/np.max(conventional_fd_optimal_aep)*100))
print("Total Improvement: {:.2f}%".format((np.max(flowers_ad_optimal_aep) - np.max(conventional_fd_optimal_aep))/np.max(conventional_fd_optimal_aep)*100))
print()
print("FLOWERS-AD Std AEP: {:.2f} GWh".format(np.std(flowers_ad_optimal_aep)/1e9))
print("FLOWERS-FD Std AEP: {:.2f} GWh".format(np.std(flowers_fd_optimal_aep)/1e9))
print("Gradient Improvement: {:.2f}%".format((np.std(flowers_ad_optimal_aep) - np.std(flowers_fd_optimal_aep))/np.std(flowers_fd_optimal_aep)*100))
print("Conventional-FD Std AEP: {:.2f} GWh".format(np.std(conventional_fd_optimal_aep)/1e9))
print("Model Improvement: {:.2f}%".format((np.std(flowers_fd_optimal_aep) - np.std(conventional_fd_optimal_aep))/np.std(conventional_fd_optimal_aep)*100))
print("Total Improvement: {:.2f}%".format((np.std(flowers_ad_optimal_aep) - np.std(conventional_fd_optimal_aep))/np.std(conventional_fd_optimal_aep)*100))
print()
print("FLOWERS-AD Median AEP Gain: {:.2f}%".format(np.median((flowers_ad_optimal_aep - initial_aep)/initial_aep*100)))
print("FLOWERS-FD Median AEP Gain: {:.2f}%".format(np.median((flowers_fd_optimal_aep - initial_aep)/initial_aep*100)))
print("Conventional-FD Median AEP Gain: {:.2f}%".format(np.median((conventional_fd_optimal_aep - initial_aep)/initial_aep*100)))
print()
print("FLOWERS-AD Median Iterations: {:.0f}".format(np.median(flowers_ad_iterations)))
print("FLOWERS-FD Median Iterations: {:.0f}".format(np.median(flowers_fd_iterations)))
print("Gradient Improvement: {:.2f}%".format((np.median(flowers_ad_iterations) - np.median(flowers_fd_iterations))/np.median(flowers_fd_iterations)*100))
print("Conventional-FD Median Iterations: {:.0f}".format(np.median(conventional_fd_iterations)))
print("Model Improvement: {:.2f}%".format((np.median(flowers_fd_iterations) - np.median(conventional_fd_iterations))/np.median(conventional_fd_iterations)*100))
print("Total Improvement: {:.2f}%".format((np.median(flowers_ad_iterations) - np.median(conventional_fd_iterations))/np.median(conventional_fd_iterations)*100))
print()
print("FLOWERS-AD Std Iterations: {:.0f}".format(np.std(flowers_ad_iterations)))
print("FLOWERS-FD Std Iterations: {:.0f}".format(np.std(flowers_fd_iterations)))
print("Gradient Improvement: {:.2f}%".format((np.std(flowers_ad_iterations) - np.std(flowers_fd_iterations))/np.std(flowers_fd_iterations)*100))
print("Conventional-FD Std Iterations: {:.0f}".format(np.std(conventional_fd_iterations)))
print("Model Improvement: {:.2f}%".format((np.std(flowers_fd_iterations) - np.std(conventional_fd_iterations))/np.std(conventional_fd_iterations)*100))
print("Total Improvement: {:.2f}%".format((np.std(flowers_ad_iterations) - np.std(conventional_fd_iterations))/np.std(conventional_fd_iterations)*100))
print()
print("FLOWERS-AD Median Time per Iteration: {:.3f} s".format(np.median(flowers_ad_tperi)))
print("FLOWERS-FD Median Time per Iteration: {:.3f} s".format(np.median(flowers_fd_tperi)))
print("Gradient Speed-Up: {:.2f}x".format(np.median(flowers_fd_tperi)/np.median(flowers_ad_tperi)))
print("Conventional-FD Median Time per Iteration: {:.3f} s".format(np.median(conventional_fd_tperi)))
print("Model Speed-Up: {:.2f}x".format(np.median(conventional_fd_tperi)/np.median(flowers_fd_tperi)))
print("Total Speed-Up: {:.2f}x".format(np.median(conventional_fd_tperi)/np.median(flowers_ad_tperi)))
print()
print("FLOWERS-AD Normalized Std Time per Iteration: {:.2f}%".format(np.std(flowers_ad_tperi)/np.mean(flowers_ad_tperi)*100))
print("FLOWERS-FD Normalized Std Time per Iteration: {:.2f}%".format(np.std(flowers_fd_tperi)/np.mean(flowers_fd_tperi)*100))
print("Gradient Improvement: {:.2f}%".format((np.std(flowers_ad_tperi)/np.mean(flowers_ad_tperi)-np.std(flowers_fd_tperi)/np.mean(flowers_fd_tperi))/(np.std(flowers_fd_tperi)/np.mean(flowers_fd_tperi))*100))
print("Conventional-FD Normalized Std Time per Iteration: {:.2f}%".format(np.std(conventional_fd_tperi)/np.mean(conventional_fd_tperi)*100))
print("Model Improvement: {:.2f}%".format((np.std(flowers_fd_tperi)/np.mean(flowers_fd_tperi)-np.std(conventional_fd_tperi)/np.mean(conventional_fd_tperi))/(np.std(conventional_fd_tperi)/np.mean(conventional_fd_tperi))*100))
print("Total Improvement: {:.2f}%".format((np.std(flowers_ad_tperi)/np.mean(flowers_ad_tperi)-np.std(conventional_fd_tperi)/np.mean(conventional_fd_tperi))/(np.std(conventional_fd_tperi)/np.mean(conventional_fd_tperi))*100))
print()
print("FLOWERS-AD Initial-Optimal Correlation: {:.2f}".format(np.corrcoef(initial_aep,flowers_ad_optimal_aep)[0,1]))
print("FLOWERS-FD Initial-Optimal Correlation: {:.2f}".format(np.corrcoef(initial_aep,flowers_fd_optimal_aep)[0,1]))
print("Conventional-FD Initial-Optimal Correlation: {:.2f}".format(np.corrcoef(initial_aep,conventional_fd_optimal_aep)[0,1]))
print("=================================================================")

# Exit codes
fig, ax = plt.subplots(1,1,figsize=(6.5,2.5))

k = []
v_fad = []
v_ffd = []
v_cfd = []

dy = 0.2
for key in flowers_ad_codes_reduced.keys():
    k.append(key)
    v_fad.append(flowers_ad_codes_reduced[key])
    v_ffd.append(flowers_fd_codes_reduced[key])
    v_cfd.append(conventional_fd_codes_reduced[key])
ax.barh(np.arange(len(k))+dy, v_fad, align='center', color=color_flowers, height=0.2,label='FLOWERS-AD',edgecolor='k',linewidth=0.75)
ax.barh(np.arange(len(k)), v_ffd, align='center', color=color_numerical, height=0.2,label='FLOWERS-FD',edgecolor='k',linewidth=0.75)
ax.barh(np.arange(len(k))-dy, v_cfd, align='center', color=color_conventional, height=0.2,label='Conventional-FD',edgecolor='k',linewidth=0.75)

ax.set(xlabel='Count',yticks=range(len(k)),yticklabels=k)
ax.legend(loc='upper right')

fig.tight_layout()
if save:
    plt.savefig(fig_name + 'codes.png', dpi=dpi)

# Superimposed layouts
fig, ax = plt.subplots(2,2,figsize=(6.5,5))
for n in range(N):
    # Initial
    solution = flowers_ad[n]
    layout_init = []
    for i in range(len(solution['init_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
    layouts_init = coll.PatchCollection(layout_init, color=color_neutral, alpha=0.2)
    ax[0][0].add_collection(layouts_init)
    ax[0][0].set_title('Initial')

    # FLOWERS-AD
    layout = []
    if flowers_ad_outliers[n] == 0:
        for i in range(len(solution['init_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_flowers, alpha=0.2)
        ax[1][0].add_collection(layouts)
    ax[1][0].set_title('FLOWERS-AD')

    # FLOWERS-FD
    solution = flowers_fd[n]
    layout = []
    if flowers_fd_outliers[n] == 0:
        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_numerical, alpha=0.2)
        ax[1][1].add_collection(layouts)
    ax[1][1].set_title('FLOWERS-FD')

    # Conventional-FD
    solution = conventional_fd[n]
    layout = []
    if conventional_fd_outliers[n] == 0:
        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_conventional, alpha=0.2)
        ax[0][1].add_collection(layouts)
    ax[0][1].set_title('Conventional-FD')

    for i in range(2):
        for j in range(2):
            ax[i][j].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
            ax[i][j].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',xticks=xticks,yticks=yticks)
            ax[i][j].grid(linestyle=':',linewidth=0.5)

fig.tight_layout(rect=[0.05,0,0.95,1])
if save:
    plt.savefig(fig_name + 'layouts.png', dpi=dpi)

# Best layouts
n_fad = np.argmax(flowers_ad_optimal_aep) # 8
n_ffd = np.argmax(flowers_fd_optimal_aep) # 8
n_cfd = np.argmax(conventional_fd_optimal_aep) # 2

# if farm == "small":
#     n_fad = np.argmax(flowers_ad_optimal_aep) # 8
#     n_ffd = np.argmax(flowers_fd_optimal_aep) # 8
#     n_cfd = np.argmax(conventional_fd_optimal_aep) # 2
# elif farm == "medium":
#     n_fad = np.argmax(flowers_ad_optimal_aep) # 25
#     n_ffd = np.argmax(flowers_fd_optimal_aep) # 25
#     n_cfd = np.argmax(conventional_fd_optimal_aep) # 56
# elif farm == "large":
#     n_fad = np.argmax(flowers_ad_optimal_aep) # 75
#     n_ffd = np.argmax(flowers_fd_optimal_aep) # 26
#     n_cfd = np.argmax(conventional_fd_optimal_aep) # 28

fig, ax = plt.subplots(1,3,figsize=(6.5,3))

# FLOWERS-AD
solution = flowers_ad[n_fad]
layout = []
for i in range(len(solution['opt_x'])):
    layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
layouts = coll.PatchCollection(layout, color=color_flowers,zorder=2)
ax[0].add_collection(layouts)
ax[0].set_title('FLOWERS-AD')

# FLOWERS-FD
solution = flowers_fd[n_ffd]
layout = []
for i in range(len(solution['opt_x'])):
    layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
layouts = coll.PatchCollection(layout, color=color_numerical,zorder=2)
ax[1].add_collection(layouts)
ax[1].set_title('FLOWERS-FD')

# Conventional-FD
solution = conventional_fd[n_cfd]
layout = []
for i in range(len(solution['opt_x'])):
    layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
layouts = coll.PatchCollection(layout, color=color_conventional,zorder=2)
ax[2].add_collection(layouts)
ax[2].set_title('Conventional-FD')

for i in range(3):
    ax[i].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
    ax[i].set(xlabel='$x/D$',ylabel='$y/D$',aspect='equal',xticks=xticks,yticks=yticks)
    ax[i].grid(linestyle=':',linewidth=0.5)

fig.tight_layout()
if save:
    plt.savefig(fig_name + 'best.png', dpi=dpi)

# Quantities of interest
fig, ax = plt.subplots(3,1,figsize=(6.5,6))

# AEP vs Time
ax[0].scatter(flowers_ad_time/time_scale,flowers_ad_optimal_aep/1e9,8,alpha=0.75,color=color_flowers,label='FLOWERS-AD',zorder=3)
ax[0].scatter(flowers_fd_time/time_scale,flowers_fd_optimal_aep/1e9,8,alpha=0.75,color=color_numerical,label='FLOWERS-FD',zorder=2)
ax[0].scatter(conventional_fd_time/time_scale,conventional_fd_optimal_aep/1e9,8,alpha=0.75,color=color_conventional,label='Conventional-FD',zorder=1)
ax[0].set(xticks=timeticks)
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
ax[0].hlines(np.median(conventional_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[0].hlines(np.median(flowers_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[0].hlines(np.median(flowers_ad_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[0].vlines(np.median(conventional_fd_time)/time_scale,ylim[0],ylim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[0].vlines(np.median(flowers_fd_time)/time_scale,ylim[0],ylim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[0].vlines(np.median(flowers_ad_time)/time_scale,ylim[0],ylim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[0].set(xlabel='Solver Time ' + time_scale_string,ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[0].grid(linestyle=':',linewidth=0.5)
# ax[0].set_axisbelow(False)

# AEP vs Iterations
ax[1].scatter(flowers_ad_iterations,flowers_ad_optimal_aep/1e9,8,alpha=0.75,color=color_flowers,label='FLOWERS-AD',zorder=3)
ax[1].scatter(flowers_fd_iterations,flowers_fd_optimal_aep/1e9,8,alpha=0.75,color=color_numerical,label='FLOWERS-FD',zorder=2)
ax[1].scatter(conventional_fd_iterations,conventional_fd_optimal_aep/1e9,8,alpha=0.75,color=color_conventional,label='Conventional-FD',zorder=1)
xlim = ax[1].get_xlim()
ylim = ax[1].get_ylim()
ax[1].hlines(np.median(conventional_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[1].hlines(np.median(flowers_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[1].hlines(np.median(flowers_ad_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[1].vlines(np.median(conventional_fd_iterations),ylim[0],ylim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[1].vlines(np.median(flowers_fd_iterations),ylim[0],ylim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[1].vlines(np.median(flowers_ad_iterations),ylim[0],ylim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[1].set(xlabel='Iterations',ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[1].grid(linestyle=':',linewidth=0.5)
# ax[1].set_axisbelow(False)

# AEP vs Time per Iterations
ax[2].scatter(flowers_ad_tperi/tperi_scale,flowers_ad_optimal_aep/1e9,8,alpha=0.75,color=color_flowers,label='FLOWERS-AD',zorder=3)
ax[2].scatter(flowers_fd_tperi/tperi_scale,flowers_fd_optimal_aep/1e9,8,alpha=0.75,color=color_numerical,label='FLOWERS-FD',zorder=2)
ax[2].scatter(conventional_fd_tperi/tperi_scale,conventional_fd_optimal_aep/1e9,8,alpha=0.75,color=color_conventional,label='Conventional-FD',zorder=1)
xlim = ax[2].get_xlim()
ylim = ax[2].get_ylim()
ax[2].hlines(np.median(conventional_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[2].hlines(np.median(flowers_fd_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[2].hlines(np.median(flowers_ad_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[2].vlines(np.median(conventional_fd_tperi)/tperi_scale,ylim[0],ylim[1],colors=color_conventional,linestyles='--',linewidths=0.75)
ax[2].vlines(np.median(flowers_fd_tperi)/tperi_scale,ylim[0],ylim[1],colors=color_numerical,linestyles='--',linewidths=0.75)
ax[2].vlines(np.median(flowers_ad_tperi)/tperi_scale,ylim[0],ylim[1],colors=color_flowers,linestyles='--',linewidths=0.75)
ax[2].set(xlabel='Solver Time per Iteration '+ tperi_scale_string,ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[2].grid(linestyle=':',linewidth=0.5)
ax[2].legend(loc='lower left')
# ax[2].set_axisbelow(False)

ax[0].text(-0.03,-0.11,'(a)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[0].transAxes)
ax[1].text(-0.03,-0.11,'(b)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[1].transAxes)
ax[2].text(-0.03,-0.11,'(c)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[2].transAxes)

# if farm == "large":
#     ax[0].text(0.86,0.04,'(a)',fontweight='bold',transform=ax[0].transAxes)
# else:
#     ax[0].text(0.92,0.04,'(a)',fontweight='bold',transform=ax[0].transAxes)
# ax[1].text(0.92,0.17,'(b)',fontweight='bold',transform=ax[1].transAxes)

fig.tight_layout()
if save:
    plt.savefig(fig_name + 'qoi.png', dpi=dpi)

# AEP Gain
fig, ax = plt.subplots(2,1,figsize=(6.5,5),sharey=True)

off_val = 0.22
mrk_sz = 1
line_width = 0.75

for n in range(N//2):
    if flowers_ad_outliers[n] == 0:
        ax[0].scatter(n-off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[0].scatter(n-off_val,flowers_ad_optimal_aep[n]/1e9,mrk_sz,color=color_flowers)
        ax[0].vlines(n-off_val,initial_aep[n]/1e9,flowers_ad_optimal_aep[n]/1e9,color=color_flowers,linewidth=line_width)
    
    if flowers_fd_outliers[n] == 0:
        ax[0].scatter(n,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[0].scatter(n,flowers_fd_optimal_aep[n]/1e9,mrk_sz,color=color_numerical)
        ax[0].vlines(n,initial_aep[n]/1e9,flowers_fd_optimal_aep[n]/1e9,color=color_numerical,linewidth=line_width)

    if conventional_fd_outliers[n] == 0:
        ax[0].scatter(n+off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[0].scatter(n+off_val,conventional_fd_optimal_aep[n]/1e9,mrk_sz,color=color_conventional)
        ax[0].vlines(n+off_val,initial_aep[n]/1e9,conventional_fd_optimal_aep[n]/1e9,color=color_conventional,linewidth=line_width)

for n in np.arange(N//2,N):
    if flowers_ad_outliers[n] == 0:
        ax[1].scatter(n-off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[1].scatter(n-off_val,flowers_ad_optimal_aep[n]/1e9,mrk_sz,color=color_flowers)
        ax[1].vlines(n-off_val,initial_aep[n]/1e9,flowers_ad_optimal_aep[n]/1e9,color=color_flowers,linewidth=line_width)
    
    if flowers_fd_outliers[n] == 0:
        ax[1].scatter(n,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[1].scatter(n,flowers_fd_optimal_aep[n]/1e9,mrk_sz,color=color_numerical)
        ax[1].vlines(n,initial_aep[n]/1e9,flowers_fd_optimal_aep[n]/1e9,color=color_numerical,linewidth=line_width)

    if conventional_fd_outliers[n] == 0:
        ax[1].scatter(n+off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
        ax[1].scatter(n+off_val,conventional_fd_optimal_aep[n]/1e9,mrk_sz,color=color_conventional)
        ax[1].vlines(n+off_val,initial_aep[n]/1e9,conventional_fd_optimal_aep[n]/1e9,color=color_conventional,linewidth=line_width)

ax[0].set(xlabel='',ylabel='AEP [GWh]',xticks=[0,10,20,30,40,50])
ax[0].grid(linestyle=':',linewidth=0.5)
ax[1].scatter([],[],10,color=color_neutral,label='Initial')
ax[1].scatter([],[],10,color=color_flowers,label='FLOWERS-AD')
ax[1].scatter([],[],10,color=color_numerical,label='FLOWERS-FD')
ax[1].scatter([],[],10,color=color_conventional,label='Conventional-FD')
ax[1].set(xlabel='Index',ylabel='AEP [GWh]',xticks=[50,60,70,80,90,100])
ax[1].legend(ncols=4,loc='lower center',bbox_to_anchor=(0.5,-0.36))
ax[1].grid(linestyle=':',linewidth=0.5)
fig.tight_layout()

# if farm == "small":
#     xyticks = [960,980,1000,1020,1040]
# elif farm == "medium":
#     xyticks = [960,980,1000,1020,1040]
# elif farm == "large":
#     xyticks = [960,980,1000,1020,1040]

# fig, ax = plt.subplots(figsize=(3.2,3))
# ax.scatter(initial_aep/1e9,flowers_ad_optimal_aep/1e9,5,alpha=0.75,color=color_flowers,label='FLOWERS-AD')
# ax.scatter(initial_aep/1e9,flowers_fd_optimal_aep/1e9,5,alpha=0.75,color=color_numerical,label='FLOWERS-FD')
# ax.scatter(initial_aep/1e9,conventional_fd_optimal_aep/1e9,5,alpha=0.75,color=color_conventional,label='Conventional-FD')
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# ax.set(xlabel='Initial AEP [GWh]',ylabel='Optimal AEP [GWh]',aspect='equal',xlim=[xlim[0],ylim[1]],ylim=[xlim[0],ylim[1]])
# ax.set(xticks=xyticks,yticks=xyticks)
# ax.legend(loc='lower right')
# ax.grid(linestyle=':',linewidth=0.5)

fig.tight_layout()
if save:
    plt.savefig(fig_name + 'aepgain.png', dpi=dpi)

# Highlight Image
if farm == "medium":

    fig, ax = plt.subplots(1,3,figsize=(8,4.5))
    n_hi = np.argsort(flowers_ad_time)[47]
    # FLOWERS-AD
    solution = flowers_ad[n_hi]
    layout_init = []
    layout_opt = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_opt.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
    layout0 = coll.PatchCollection(layout_init, linestyle='--', linewidth=0.75, facecolor='none', edgecolor=color_flowers,zorder=2)
    layout1 = coll.PatchCollection(layout_opt, color=color_flowers,zorder=2)
    ax[0].text(0.36,1.02,'FLOWERS-AD',color=color_flowers,transform=ax[0].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[0].text(0.5,-0.07,'+5.2% AEP in 1.8 s',transform=ax[0].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[0].add_collection(layout0)
    ax[0].add_collection(layout1)

    # FLOWERS-FD
    solution = flowers_fd[n_hi]
    layout_init = []
    layout_opt = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_opt.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
    layout0 = coll.PatchCollection(layout_init, linestyle='--', linewidth=0.75, facecolor='none', edgecolor=color_numerical,zorder=2)
    layout1 = coll.PatchCollection(layout_opt, color=color_numerical,zorder=2)
    ax[1].text(0.36,1.02,'FLOWERS-FD',color=color_numerical,transform=ax[1].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[1].text(0.5,-0.07,'+5.2% AEP in 29 s',transform=ax[1].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[1].add_collection(layout0)
    ax[1].add_collection(layout1)

    # Conventional-FD
    solution = conventional_fd[n_hi]
    layout_init = []
    layout_opt = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_opt.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
    layout0 = coll.PatchCollection(layout_init, linestyle='--', linewidth=0.75, facecolor='none', edgecolor=color_conventional,zorder=2)
    layout1 = coll.PatchCollection(layout_opt, color=color_conventional,zorder=2)
    ax[2].text(0.36,1.02,'Conventional-FD',color=color_conventional,transform=ax[2].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[2].text(0.5,-0.07,'+4.6% AEP in 45 min',transform=ax[2].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=11,fontweight='bold')
    ax[2].add_collection(layout0)
    ax[2].add_collection(layout1)

    for i in range(3):
        ax[i].fill(boundaries[0],boundaries[1],color='whitesmoke',zorder=1)
        ax[i].plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
        ax[i].set_aspect('equal')
        ax[i].axis('off')

    fig.text(0.05,0.90,r'$\langle$AEP Model$\rangle$ – $\langle$Gradient Source$\rangle$',transform=fig.transFigure,horizontalalignment='left',verticalalignment='center',fontsize=11)
    fig.text(0.05,0.835,'FLOWERS',transform=fig.transFigure,horizontalalignment='left',verticalalignment='center',fontsize=8)
    fig.text(0.05,0.784,'Conventional',transform=fig.transFigure,horizontalalignment='left',verticalalignment='center',fontsize=8)
    fig.text(0.355,0.835,'Analytic Derivatives (AD)',transform=fig.transFigure,horizontalalignment='right',verticalalignment='center',fontsize=8)
    fig.text(0.355,0.784,'Finite Differences (FD)',transform=fig.transFigure,horizontalalignment='right',verticalalignment='center',fontsize=8)

    fig.text(0.75,0.835,'Medium Case',transform=fig.transFigure,horizontalalignment='center',verticalalignment='center',fontsize=8)
    fig.text(0.75,0.784,'50 turbines',transform=fig.transFigure,horizontalalignment='center',verticalalignment='center',fontsize=8)

    tmp0 = plt.scatter([],[],25,marker='o',color='k',facecolor='none',linestyle='--',label='Initial',linewidth=0.75)
    tmp1 = plt.scatter([],[],25,marker='o',color='k',label='Optimal')
    fig.legend(handles=[tmp0,tmp1],bbox_to_anchor=(0.65,0.868),frameon=False,fontsize=8,labelspacing=1)
    fig.tight_layout(rect=(0.,0.,1.,0.8))

    axr = fig.add_axes([0.83,0.72,0.14,0.25],projection='polar')
    vis.plot_wind_rose(tl.load_wind_rose(1), ax=axr)
    axr.axis('off')
    axr.text(0,0.06,'N',fontsize=6,horizontalalignment='center',verticalalignment='center',fontweight='bold')
    axr.get_legend().remove()

    if save:
        plt.savefig('./figures/opt_highlight.jpg', dpi=1200)

# ###########################################################################
# # FLOWERS PARAMETER SENSITIVITY
# ###########################################################################

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

# Load files
N = 100

solutions_parameter = [list() for _ in range(10)]

cc = co.ColorConverter.to_rgb(color_flowers)
color_parameter = [scale_lightness(cc, scale) for scale in [1.8,1.6,1.4,1.2,1,0.83,0.71,0.625,0.55,0.45]]
cmap = co.LinearSegmentedColormap.from_list('colors',color_parameter,N=10)

file_base = 'solutions/opt_' + farm + '_parameter_'
fig_name = './figures/opt_parameter_' + farm + '_'

for k_idx, k in enumerate([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]):
    for n in range(N):
        if k == 0.05:
            file_name = 'solutions/opt_' + farm + '_flowers_analytical_' + str(n) + '.p'
        else:
            file_name = file_base + '{:.2f}'.format(k) + '_' + str(n) + '.p'  
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_parameter[k_idx].append(solution)

boundaries /= 126.
boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

# Collect statistics
initial_aep = np.zeros(N)
parameter_optimal_aep = np.zeros((10,N))
parameter_time = np.zeros((10,N))
parameter_iterations = np.zeros((10,N))

for n in range(N):
    initial_aep[n] = solutions_parameter[0][n]['init_aep']
    for k in range(10):
        parameter_optimal_aep[k,n] = solutions_parameter[k][n]['opt_aep']
        parameter_time[k,n] = solutions_parameter[k][n]['total_time']
        parameter_iterations[k,n] = solutions_parameter[k][n]['iter']

parameter_tperi = parameter_time / parameter_iterations

# Best cases
fig, ax = plt.subplots(1,1,figsize=(6.5,3))
nn = np.argmax(parameter_optimal_aep,axis=1)

if save:
    print(np.mean(np.max(parameter_optimal_aep,axis=1))/1e9)
    print((np.max(np.max(parameter_optimal_aep,axis=1)) - np.mean(np.max(parameter_optimal_aep,axis=1)))/1e9)
    print((np.mean(np.max(parameter_optimal_aep,axis=1)) - np.min(np.max(parameter_optimal_aep,axis=1)))/1e9)

# if farm == "small":
#     nn = np.argmax(parameter_optimal_aep,axis=0) # [20,80,50,24,8,55,7,55,95,83]
#     case_title = 'Best: 251.8 $\pm$ 0.5 GWh'
# elif farm == "medium":
#     nn = # [33,25,85,1,25,58,58,32,25,25]
#     # case_title = 'Best: 1037.9 $\pm$ 0.4 GWh'
#     case_title = ''
# elif farm == "large":
#     nn = # [16,49,49,1,75,1,1,1,1,1]
#     case_title = 'Best: 4905.0 $\pm$ 116 GWh'

# # DELETE POST-PROCESSING ROUTINE
# import floris.tools as wfct
# from scipy.interpolate import NearestNDInterpolator

# post_processing = wfct.floris_interface.FlorisInterface("./input/post.yaml")
# wind_rose = tl.resample_wind_speed(tl.load_wind_rose(1), ws=np.arange(1.,26.,1.))
# wd_array = np.array(wind_rose["wd"].unique(), dtype=float)
# ws_array = np.array(wind_rose["ws"].unique(), dtype=float)
# wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
# freq_interp = NearestNDInterpolator(wind_rose[["wd", "ws"]],wind_rose["freq_val"])
# freq = freq_interp(wd_grid, ws_grid)
# _freq_2D = freq / np.sum(freq)
# post_processing.reinitialize(wind_directions=wd_array,wind_speeds=ws_array,time_series=False)
# post_aep = np.zeros((10,10)) # k,TI
# ti_list = [0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15]

for k in range(9,-1,-1):
    solution = solutions_parameter[k][nn[k]]
    layout = []
    for i in range(len(solution['opt_x'])):
        layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))
    layouts = coll.PatchCollection(layout, color=color_parameter[k], alpha=0.5, zorder=2)
    ax.add_collection(layouts)

    # # DELETE POST-PROCESSING
    # post_processing.reinitialize(layout_x=solution["opt_x"].flatten(),layout_y=solution["opt_y"].flatten(),time_series=False)
    # for TI_idx, TI in enumerate(ti_list):
    #     print(TI_idx)
    #     post_processing.reinitialize(turbulence_intensity=TI,time_series=False)
    #     post_processing.calculate_wake()
    #     post_aep[k,TI_idx] = np.sum(post_processing.get_farm_power() * _freq_2D * 8760)

# # DELETE POST-PROCESSING
# for TI_idx, TI in enumerate(ti_list):
#     print(TI)
#     print(np.mean(post_aep[:,TI_idx])/1e9)
#     print((np.max(post_aep[:,TI_idx]) - np.mean(post_aep[:,TI_idx]))/np.mean(post_aep[:,TI_idx])*100)
#     print((np.mean(post_aep[:,TI_idx]) - np.min(post_aep[:,TI_idx]))/np.mean(post_aep[:,TI_idx])*100)

ax.plot(boundaries[0],boundaries[1],color='k',linewidth=1.5,zorder=1)
ax.set(xlabel='$x/D$',ylabel='$y/D$',aspect='equal',xticks=xticks,yticks=yticks)
ax.grid(linestyle=':',linewidth=0.5)

cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=co.Normalize(vmin=0.01,vmax=0.11)),ax=ax,label='$k$', ticks=[0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095,0.105],fraction=0.07,shrink=0.95)
cbar.set_ticklabels(['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.10'])
fig.tight_layout(rect=[0.2,0,0.8,1])
if save:
    plt.savefig(fig_name + 'layouts.png', dpi=dpi)

# Quantities of interest
fig, ax = plt.subplots(3,1,figsize=(6.5,6))

# AEP vs Time
for k in range(9,-1,-1):
    ax[0].scatter(parameter_time[k],parameter_optimal_aep[k]/1e9,8,alpha=0.5,color=color_parameter[k])
# ax[0].set(xticks=timeticks)
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
for k in range(9,-1,-1):
    ax[0].hlines(np.median(parameter_optimal_aep[k])/1e9,xlim[0],xlim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
    ax[0].vlines(np.median(parameter_time[k]),ylim[0],ylim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
ax[0].set(xlabel='Solver Time [s]',ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[0].grid(linestyle=':',linewidth=0.5)
# ax[0].set_axisbelow(False)

# AEP vs Iterations
for k in range(9,-1,-1):
    ax[1].scatter(parameter_iterations[k],parameter_optimal_aep[k]/1e9,8,alpha=0.5,color=color_parameter[k])
# ax[1].set(xticks=timeticks)
xlim = ax[1].get_xlim()
ylim = ax[1].get_ylim()
for k in range(9,-1,-1):
    ax[1].hlines(np.median(parameter_optimal_aep[k])/1e9,xlim[0],xlim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
    ax[1].vlines(np.median(parameter_iterations[k]),ylim[0],ylim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
ax[1].set(xlabel='Iterations',ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[1].grid(linestyle=':',linewidth=0.5)
# ax[1].set_axisbelow(False)

# AEP vs Time per Iterations
for k in range(9,-1,-1):
    ax[2].scatter(parameter_tperi[k],parameter_optimal_aep[k]/1e9,8,alpha=0.5,color=color_parameter[k])
ax[2].set(xticks=[0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07])
xlim = ax[2].get_xlim()
ylim = ax[2].get_ylim()
for k in range(9,-1,-1):
    ax[2].hlines(np.median(parameter_optimal_aep[k])/1e9,xlim[0],xlim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
    ax[2].vlines(np.median(parameter_tperi[k]),ylim[0],ylim[1],colors=color_parameter[k],linestyles='--',linewidths=0.75)
ax[2].set(xlabel='Solver Time per Iteration [s]',ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
ax[2].grid(linestyle=':',linewidth=0.5)
# ax[2].set_axisbelow(False)

fig.tight_layout()
fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.89, 0.197, 0.02, 0.655])
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=co.Normalize(vmin=0.01,vmax=0.11)),cax=cbar_ax,label='$k$', ticks=[0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095,0.105],fraction=0.07,pad=0.02,shrink=0.85)
cbar.set_ticklabels(['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.10'])
ax[0].text(-0.08,-0.11,'(a)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[0].transAxes)
ax[1].text(-0.08,-0.11,'(b)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[1].transAxes)
ax[2].text(-0.08,-0.11,'(c)',fontweight='bold',horizontalalignment='center',verticalalignment='center',transform=ax[2].transAxes)

if save:
    plt.savefig(fig_name + 'qoi.png', dpi=dpi)

plt.show()