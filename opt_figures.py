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

save = True
dpi = 500
figs = "parameter"

###########################################################################
# FLOWERS MULTISTART
###########################################################################

if figs == "multistart":
    # Load files
    N = 25 # idx = 8 outlier
    solutions = []
    for n in range(N):
        file_name = 'solutions/opt_multi' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions.append(solution)

    boundaries /= 126.
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

    vis.plot_wind_rose(wr)

    fig, ax = plt.subplots(1,1)
    # aep_cmap = cm.get_cmap('winter')
    # nn = np.linspace(0.,1.,N,endpoint=True)
    for n in range(N):
        solution = solutions[n]
        ax.plot(range(solution['iter']),solution['hist_aep']/1e9,color='tab:blue',alpha=0.7)
    ax.set(xlabel='Iteration', ylabel='AEP [GWh]')
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_history.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        layout = []
        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color='tab:blue', alpha=0.5)
        ax.add_collection(layouts)

    ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax.set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Optimal Layouts')
    ax.grid(True)
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_opt_layouts.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        layout = []
        for i in range(len(solution['init_x'])):
            layout.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color='tab:orange', alpha=0.5)
        ax.add_collection(layouts)

    ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax.set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Initial Layouts')
    ax.grid(True)
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_init_layouts.png', dpi=dpi)

    # 16 is optimal case
    fig, ax = plt.subplots(1,1)
    nn = 16
    solution = solutions[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color='tab:orange',label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color='tab:blue',label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color='tab:orange')
    layout1 = coll.PatchCollection(layout_final, color='tab:blue')
    ax.add_collection(layout0)
    ax.add_collection(layout1)
    ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax.set(xlabel='x/D', ylabel='y/D', aspect='equal',title='Case ' + str(nn))
    ax.grid(True)
    ax.legend(handles=[tmp0,tmp1],loc='upper right')
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_success.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        ax.scatter(solution['total_time'],solution['opt_aep']/1e9,20,color='tab:blue')
    ax.set(xlabel='Solver Time [s]',ylabel='Optimal AEP [GWh]')
    # print(ax.get_ylim())
    if save:
        plt.savefig('./figures/opt_multistart_aepvstime.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        ax.scatter(solution['iter'],solution['opt_aep']/1e9,20,color='tab:blue')
    ax.set(xlabel='Iterations',ylabel='Optimal AEP [GWh]')
    if save:
        plt.savefig('./figures/opt_multistart_aepvsiter.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        ax.scatter(solution['init_aep']/1e9,solution['opt_aep']/1e9,20,color='tab:blue')
    ax.set(xlabel='Initial AEP [GWh]',ylabel='Optimal AEP [GWh]')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmin = np.min([xlim[0],ylim[0]])
    xmax = np.max([xlim[1],ylim[1]])
    ax.set(xlim=[xmin,xmax], ylim=[xmin,xmax])
    if save:
        plt.savefig('./figures/opt_multistart_aepvsaep.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(N):
        solution = solutions[n]
        ax.scatter(solution['iter'],solution['total_time'],20,color='tab:blue')
    ax.set(xlabel='Iterations',ylabel='Solver Time [s]')
    if save:
        plt.savefig('./figures/opt_multistart_timevsiter.png', dpi=dpi)

    fig, ax = plt.subplots(1,1,figsize=(13,3))
    exit_codes = {}
    for n in range(N):
        exit_code = solutions[n]['exit_code']
        if exit_code in exit_codes:
            exit_codes[exit_code] += 1
        else:
            exit_codes[exit_code] = 1

    ax.bar(range(len(exit_codes)), list(exit_codes.values()), align='center',color='tab:blue',width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(exit_codes)), xticklabels=list(exit_codes.keys()))
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_codes.png', dpi=dpi)

    optimal_aep = np.zeros(N)
    solver_time = np.zeros(N)
    for n in range(N):
        optimal_aep[n] = solutions[n]['opt_aep']
        solver_time[n] = solutions[n]['total_time']

    print("Median Optimal AEP: {:.2f} GWh".format(np.median(optimal_aep)/1e9))
    print("Median Solver Time: {:.2f} s".format(np.median(solver_time)))

    plt.show()

###########################################################################
# FLOWERS PARAMETER SENSITIVITY
###########################################################################

if figs == "parameter":
    # Load files
    k = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])
    solutions = []

    for idx in range(len(k)):
        file_name = 'solutions/opt_k' + str(k[idx])[-2:] + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions.append(solution)

    boundaries /= 126.
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)


    fig, ax = plt.subplots(1,1)
    for n in range(len(k)):
        solution = solutions[n]
        ax.plot(range(solution['iter']),solution['hist_aep']/1e9,color='tab:blue',alpha=0.7)
    ax.set(xlabel='Iteration', ylabel='AEP [GWh]')
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_parameter_history.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(len(k)):
        solution = solutions[n]
        layout = []
        # if n == 1:
        #     continue
        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color='tab:blue', alpha=0.5)
        ax.add_collection(layouts)

    ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax.set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Optimal Layouts')
    ax.grid(True)
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_parameter_layouts.png', dpi=dpi)

    fig, ax = plt.subplots(2,5,figsize=(14,7))
    for nn in range(len(k)):
        axx = int(nn >= 5)
        axy = int(nn % 5)
        solution = solutions[nn]
        layout_init = []
        layout_final = []
        for i in range(len(solution['opt_x'])):
            layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
            layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        tmp0 = plt.Circle(([],[]),1/2,color='tab:orange',label='Initial Layout')
        tmp1 = plt.Circle(([],[]),1/2,color='tab:blue',label='Optimal Layout')

        layout0 = coll.PatchCollection(layout_init, color='tab:orange')
        layout1 = coll.PatchCollection(layout_final, color='tab:blue')
        ax[axx,axy].add_collection(layout0)
        ax[axx,axy].add_collection(layout1)
        ax[axx,axy].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
        ax[axx,axy].set(xlabel='x/D', ylabel='y/D', aspect='equal',title='$k$ = ' + str(k[nn]))
        ax[axx,axy].grid(True)
        # ax[axx,axy].legend(handles=[tmp0,tmp1],loc='upper right')
        fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_parameter_layoutsidx.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(len(k)):
        solution = solutions[n]
        ax.scatter(k[n],solution['opt_aep']/1e9,20,color='tab:blue')
    # xlim = ax.get_xlim()
    # ax.hlines(solution['init_aep']/1e9,xlim[0],xlim[1],color='tab:blue',linestyles='--')
    ax.set(xlabel='$k$',ylabel='Optimal AEP [GWh]', ylim=[388.60,392.09])
    if save:
        plt.savefig('./figures/opt_parameter_aep.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(len(k)):
        solution = solutions[n]
        ax.scatter(k[n],solution['total_time'],20,color='tab:blue')
    ax.set(xlabel='$k$',ylabel='Solver Time [s]')
    if save:
        plt.savefig('./figures/opt_parameter_time.png', dpi=dpi)

    fig, ax = plt.subplots(1,1)
    for n in range(len(k)):
        solution = solutions[n]
        ax.scatter(k[n],solution['iter'],20,color='tab:blue')
    ax.set(xlabel='$k$',ylabel='Iterations')
    if save:
        plt.savefig('./figures/opt_parameter_iter.png', dpi=dpi)

    fig, ax = plt.subplots(1,1,figsize=(13,3))
    exit_codes = {}
    for n in range(len(k)):
        exit_code = solutions[n]['exit_code']
        if exit_code in exit_codes:
            exit_codes[exit_code] += 1
        else:
            exit_codes[exit_code] = 1

    ax.bar(range(len(exit_codes)), list(exit_codes.values()), align='center',color='tab:blue',width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(exit_codes)), xticklabels=list(exit_codes.keys()))
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_parameter_codes.png', dpi=dpi)

    # optimal_aep = np.zeros(len(k))
    # solver_time = np.zeros(len(k))
    # for n in range(2):
    #     optimal_aep[n] = solutions[n]['opt_aep']
    #     solver_time[n] = solutions[n]['total_time']

    # print("Median Optimal AEP: {:.2f} GWh".format(np.median(optimal_aep)/1e9))
    # print("Median Solver Time: {:.2f} s".format(np.median(solver_time)))

    plt.show()