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
figs = "multistart"

color_neutral = 'goldenrod'
color_flowers = 'dodgerblue'
color_conventional = 'indianred'

if figs == "cases":
    # Farm definitions
    fig = plt.figure(figsize=(11,7))
    ax0 = fig.add_subplot(2,3,1)
    ax1 = fig.add_subplot(2,3,2)
    ax2 = fig.add_subplot(2,3,3)
    ax3 = fig.add_subplot(2,3,4, projection='polar')
    ax4 = fig.add_subplot(2,3,5, projection='polar')
    ax5 = fig.add_subplot(2,3,6, projection='polar')
    ax = [[ax0,ax1,ax2],[ax3,ax4,ax5]]

    # Small farm
    wr = tl.load_wind_rose(8)
    boundaries = [(0., 0.),(10, 0.),(10, 10),(0., 10)]
    xx, yy = tl.random_layout(boundaries, n_turb=10, idx=100, D=1)
    boundaries = np.array(boundaries).T
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

    layout = []
    for i in range(len(xx)):
        layout.append(plt.Circle((xx[i], yy[i]), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral)
        ax[0][0].add_collection(layouts)

    ax[0][0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][0].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='$N$ = 10')
    ax[0][0].grid(True)

    vis.plot_wind_rose(wr, ax=ax[1][0])
    ax[1][0].get_legend().remove()

    # Medium farm
    wr = tl.load_wind_rose(1)
    boundaries = [(8, 0.),(28, 0.),(36, 24),(24, 36),(0, 36)]
    xx, yy = tl.random_layout(boundaries, n_turb=50, idx=99, D=1)
    boundaries = np.array(boundaries).T
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)
    layout = []
    for i in range(len(xx)):
        layout.append(plt.Circle((xx[i], yy[i]), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral)
        ax[0][1].add_collection(layouts)

    ax[0][1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][1].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='$N$ = 50')
    ax[0][1].grid(True)

    vis.plot_wind_rose(wr, ax=ax[1][1])
    ax[1][1].get_legend().remove()

    # Large farm
    wr = tl.load_wind_rose(6)
    boundaries = [(10, 0.),(125, 0.),(125, 50),(110, 160),(40., 160),(40, 120),(0, 100),(12, 40),(15, 20)]
    xx, yy = tl.random_layout(boundaries, n_turb=250, idx=100, D=1)
    boundaries = np.array(boundaries).T
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)
    layout = []
    for i in range(len(xx)):
        layout.append(plt.Circle((xx[i], yy[i]), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral)
        ax[0][2].add_collection(layouts)

    ax[0][2].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][2].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='$N$ = 250')
    ax[0][2].grid(True)

    vis.plot_wind_rose(wr, ax=ax[1][2])
    ax[1][2].get_legend().remove()

    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_setup.png', dpi=dpi)

    plt.show()

###########################################################################
# MULTISTART
###########################################################################

if figs == "multistart":
    # Load files
    N = 50
    solutions_flowers = []
    solutions_conventional = []
    for n in range(N):
        file_name = 'solutions/opt_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_flowers.append(solution)

        file_name = 'solutions/opt_conventional_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_conventional.append(solution)

    boundaries /= 126.
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

    # Define outliers here after analyzing results
    flowers_outliers = np.zeros(N)
    flowers_outliers[[40]] = 1

    conventional_outliers = np.zeros(N)
    conventional_outliers[[3,21,25,36]] = 1

    flowers_min_dist = np.zeros(N)
    conventional_min_dist = np.zeros(N)

    # fig, ax = plt.subplots(1,1)
    # # aep_cmap = cm.get_cmap('winter')
    # # nn = np.linspace(0.,1.,N,endpoint=True)
    # for n in range(N):
    #     solution = solutions_flowers[n]
    #     ax.plot(range(solution['iter']),solution['hist_aep']/1e9,color=color_flowers,alpha=0.7)
    # ax.set(xlabel='Iteration', ylabel='AEP [GWh]')
    # fig.tight_layout()
    # if save:
    #     plt.savefig('./figures/opt_multistart_history.png', dpi=dpi)

    # Superimposed layouts
    fig, ax = plt.subplots(1,3,figsize=(11,4.5))
    for n in range(N):
        solution = solutions_flowers[n]
        layout = []
        for i in range(len(solution['init_x'])):
            layout.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral, alpha=0.5)
        ax[0].add_collection(layouts)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Initial Layouts')
    ax[0].grid(True)

    for n in range(N):
        solution = solutions_flowers[n]
        layout = []

        # # Calculate minimum distance
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # flowers_min_dist[n] = np.min(dd)

        if flowers_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_flowers, alpha=0.5)
        ax[1].add_collection(layouts)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='FLOWERS Layouts')
    ax[1].grid(True)

    for n in range(N):
        solution = solutions_conventional[n]
        layout = []

        # # Calculate minimum distance
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # conventional_min_dist[n] = np.min(dd)

        if conventional_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_conventional, alpha=0.5)
        ax[2].add_collection(layouts)
    ax[2].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[2].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Conventional Layouts')
    ax[2].grid(True)

    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_layouts.png', dpi=dpi)

    # print(np.where(flowers_min_dist < 1))
    # print(np.where(conventional_min_dist < 1))

    # Best cases
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    nn = 9
    solution = solutions_flowers[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_flowers,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_flowers)
    ax[0].add_collection(layout0)
    ax[0].add_collection(layout1)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='x/D', ylabel='y/D', aspect='equal',title='FLOWERS Best: {:.2f} GWh'.format(solution['opt_aep']/1e9))
    ax[0].grid(True)

    nn = 6
    solution = solutions_conventional[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_conventional,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_conventional)
    ax[1].add_collection(layout0)
    ax[1].add_collection(layout1)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='x/D', ylabel='y/D', aspect='equal',title='Conventional Best: {:.2f} GWh'.format(solution['opt_aep']/1e9))
    ax[1].grid(True)
    # ax.legend(handles=[tmp0,tmp1],loc='upper right')
    fig.tight_layout()
    if save:
        plt.savefig('./figures/opt_multistart_best.png', dpi=dpi)

    # Collect statistics
    initial_aep = np.zeros(N)
    flowers_optimal_aep = np.zeros(N)
    flowers_solver_time = np.zeros(N)
    flowers_iterations = np.zeros(N)
    conventional_optimal_aep = np.zeros(N)
    conventional_solver_time = np.zeros(N)
    conventional_iterations = np.zeros(N)
    for n in range(N):
        initial_aep[n] = solutions_flowers[n]['init_aep']
        flowers_optimal_aep[n] = solutions_flowers[n]['opt_aep']
        flowers_solver_time[n] = solutions_flowers[n]['total_time']
        flowers_iterations[n] = solutions_flowers[n]['iter']
        conventional_optimal_aep[n] = solutions_conventional[n]['opt_aep']
        conventional_solver_time[n] = solutions_conventional[n]['total_time']
        conventional_iterations[n] = solutions_conventional[n]['iter']
    
    print("FLOWERS Aggregate Cost: {:.2f} core-hrs".format(np.sum(flowers_solver_time)/3600))
    print("Conventional Aggregate Cost: {:.2f} core-hrs".format(np.sum(conventional_solver_time)/3600))

    # Mask outliers
    flowers_optimal_aep = np.ma.masked_where(flowers_outliers,flowers_optimal_aep)
    # flowers_solver_time = np.ma.masked_where(flowers_outliers,flowers_solver_time)
    # flowers_iterations = np.ma.masked_where(flowers_outliers,flowers_iterations)

    conventional_optimal_aep = np.ma.masked_where(conventional_outliers,conventional_optimal_aep)
    # conventional_solver_time = np.ma.masked_where(conventional_outliers,conventional_solver_time)
    # conventional_iterations = np.ma.masked_where(conventional_outliers,conventional_iterations)

    # AEP vs. Time
    fig, ax = plt.subplots(1,1)

    ax.scatter(flowers_solver_time/3600,flowers_optimal_aep/1e9,20,color=color_flowers)
    ax.scatter(conventional_solver_time/3600,conventional_optimal_aep/1e9,20,color=color_conventional)

    ax.set(xscale='log')
    plt.autoscale(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax.vlines(np.median(flowers_solver_time)/3600,ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax.hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--')
    ax.vlines(np.median(conventional_solver_time)/3600,ylim[0],ylim[1],colors=color_conventional,linestyles='--')
    ax.set(xlabel='Solver Time [hr]',ylabel='Optimal AEP [GWh]')
    if save:
        plt.savefig('./figures/opt_multistart_aepvstime.png', dpi=dpi)

    # AEP vs. Iterations
    fig, ax = plt.subplots(1,1)

    ax.scatter(flowers_iterations,flowers_optimal_aep/1e9,20,color=color_flowers)
    ax.scatter(conventional_iterations,conventional_optimal_aep/1e9,20,color=color_conventional)

    plt.autoscale(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax.vlines(np.median(flowers_iterations),ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax.hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--')
    ax.vlines(np.median(conventional_iterations),ylim[0],ylim[1],colors=color_conventional,linestyles='--')
    ax.set(xlabel='Iterations',ylabel='Optimal AEP [GWh]')
    if save:
        plt.savefig('./figures/opt_multistart_aepvsiter.png', dpi=dpi)

    # AEP Gain per Study
    fig, ax = plt.subplots(1,1,figsize=(11,4.5))
    off_val = 0.175
    mrk_sz = 8
    for n in range(N):
        if flowers_outliers[n] == 0:
            solution = solutions_flowers[n]
            ax.scatter(n-off_val,solution['init_aep']/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n-off_val,solution['opt_aep']/1e9,mrk_sz,color=color_flowers)
            ax.vlines(n-off_val,solution['init_aep']/1e9,solution['opt_aep']/1e9,color=color_flowers,linewidth=2)

        if conventional_outliers[n] == 0:
            solution = solutions_conventional[n]
            ax.scatter(n+off_val,solution['init_aep']/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n+off_val,solution['opt_aep']/1e9,mrk_sz,color=color_conventional)
            ax.vlines(n+off_val,solution['init_aep']/1e9,solution['opt_aep']/1e9,color=color_conventional,linewidth=2)
    ax.scatter([],[],20,color=color_neutral,label='Initial')
    ax.scatter([],[],20,color=color_flowers,label='FLOWERS')
    ax.scatter([],[],20,color=color_conventional,label='Conventional')
    ax.set(xlabel='Index',ylabel='AEP [GWh]')
    ax.legend()
    fig.tight_layout()

    if save:
        plt.savefig('./figures/opt_multistart_aepgain.png', dpi=dpi)

    # fig, ax = plt.subplots(1,1)
    # for n in range(N):
    #     solution = solutions_flowers[n]
    #     ax.scatter(solution['iter'],solution['total_time'],20,color=color_flowers)
    # ax.set(xlabel='Iterations',ylabel='Solver Time [s]')
    # if save:
    #     plt.savefig('./figures/opt_multistart_timevsiter.png', dpi=dpi)

    fig, ax = plt.subplots(1,1,figsize=(13,3))
    exit_codes = {}
    for n in range(N):
        exit_code = solutions_flowers[n]['exit_code']
        if exit_code in exit_codes:
            exit_codes[exit_code] += 1
        else:
            exit_codes[exit_code] = 1
    ax.bar(range(len(exit_codes)), list(exit_codes.values()), align='center',color=color_flowers,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(exit_codes)), xticklabels=list(exit_codes.keys()))
    fig.tight_layout()

    fig, ax = plt.subplots(1,1,figsize=(13,3))
    exit_codes = {}
    for n in range(N):
        exit_code = solutions_conventional[n]['exit_code']
        if exit_code in exit_codes:
            exit_codes[exit_code] += 1
        else:
            exit_codes[exit_code] = 1
    ax.bar(range(len(exit_codes)), list(exit_codes.values()), align='center',color=color_conventional,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(exit_codes)), xticklabels=list(exit_codes.keys()))
    fig.tight_layout()

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