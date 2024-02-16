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
dpi = 500

# figs = "cases"
# figs = "multistart"
# figs = "gradients"
figs = "parameter"

# farm = "small"
farm = "medium"
# farm = "large"

color_neutral = 'goldenrod'
color_flowers = 'dodgerblue'
color_conventional = 'indianred'
color_numerical = 'mediumpurple'

if figs == "cases":
    # Farm definitions
    fig = plt.figure(figsize=(11,9))
    ax0 = fig.add_subplot(2,3,1)
    ax1 = fig.add_subplot(2,3,2)
    ax2 = fig.add_subplot(2,3,3)
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
    ax[0][0].add_collection(layouts)

    ax[0][0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][0].set(xlabel='x/D', ylabel='y/D', aspect='equal', title='$N$ = 10', xticks = [0,2,4,6,8,10], yticks = [0,2,4,6,8,10])
    ax[0][0].grid(linestyle=':')
    ax[0][0].set_axisbelow(True)

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
    ax[0][1].add_collection(layouts)

    ax[0][1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N$ = 50', xticks = [0,5,10,15,20,25,30,35], yticks = [0,5,10,15,20,25,30,35])
    ax[0][1].grid(linestyle=':')
    ax[0][1].set_axisbelow(True)

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

    ax[0][2].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0][2].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='$N$ = 250', xticks = [0,25,50,75,100,125,150], yticks = [0,25,50,75,100,125,150])
    ax[0][2].grid(linestyle=':')
    ax[0][2].set_axisbelow(True)

    vis.plot_wind_rose(wr, ax=ax[1][2])
    h, l = ax[1][2].get_legend_handles_labels()
    h = h[:5]
    l = l[:5]
    ax[1][2].get_legend().remove()

    # fig.tight_layout(rect=[0.,0.04,0.,1])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05)
    fig.legend(reversed(h), l, ncol=5,loc='lower center')
    if save:
        plt.savefig('./figures/opt_multistart_setup.png', dpi=dpi)

    # fig, ax = plt.subplots(1,1)
    # for n in range(100):
    #     init_x, init_y, boundaries = tl.load_layout(n, "large")
    #     boundaries = np.array(boundaries).T
    #     boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)/126.
    #     layout = []
    #     for i in range(len(init_x)):
    #         layout.append(plt.Circle((init_x[i]/126., init_y[i]/126.), 1/2))
    #     layouts = coll.PatchCollection(layout, color=color_neutral, alpha=0.5)
    #     ax.add_collection(layouts)
    # ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    # ax.set(xlabel='x/D', ylabel='y/D', aspect='equal', title='Initial Layouts')
    # ax.grid(True)

    N = 100
    flowers_solver_time = np.zeros((3,N))
    conventional_solver_time = np.zeros((3,N))
    conventional_outliers = np.zeros((3,N))
    NT = [10,50,250]

    conventional_outliers[0,[57,60]] = 1
    conventional_outliers[1,13] = 1
    conventional_outliers[2,[18,77]] = 1

    for n in range(N):
        file_name = 'solutions/opt_small_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        flowers_solver_time[0,n] = solution['total_time']

        file_name = 'solutions/opt_small_conventional_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        conventional_solver_time[0,n] = solution['total_time']

        file_name = 'solutions/opt_medium_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        flowers_solver_time[1,n] = solution['total_time']

        file_name = 'solutions/opt_medium_conventional_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        conventional_solver_time[1,n] = solution['total_time']

        file_name = 'solutions/opt_large_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        flowers_solver_time[2,n] = solution['total_time']

        file_name = 'solutions/opt_large_conventional_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        conventional_solver_time[2,n] = solution['total_time']

    conventional_solver_time = np.ma.masked_where(conventional_outliers, conventional_solver_time)
    flowers_mean_time = np.mean(flowers_solver_time,axis=1)/60
    conventional_mean_time = np.mean(conventional_solver_time,axis=1)/60

    fig, ax = plt.subplots(1,1, figsize=(11,5))
    ax.plot(NT,flowers_mean_time,'-o',markersize=5,color=color_flowers,linewidth=2,label='FLOWERS')
    ax.plot(NT,conventional_mean_time,'-o',markersize=5,color=color_conventional,linewidth=2,label='Conventional',zorder=10)
    # bplot_c = ax.boxplot(conventional_solver_time.T/60,positions=NT,patch_artist=True)
    # for patch in bplot_c['boxes']:
    #     patch.set_facecolor(color_conventional)
    #     patch.set_alpha(0.5)
    ax.set(xlabel='Number of Turbines, $N$', ylabel='Solver Time [min]', xticks=NT)
    if save:
        plt.savefig('./figures/opt_multistart_costs.png', dpi=dpi)

    plt.show()

###########################################################################
# MULTISTART
###########################################################################

if figs == "multistart":

    # Load files
    N = 100

    solutions_flowers = []
    solutions_conventional = []

    file_base = 'solutions/opt_' + farm
    fig_name = './figures/opt_' + farm + '_'

    for n in range(N):
        file_name = file_base + '_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_flowers.append(solution)

        file_name = file_base + '_conventional_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_conventional.append(solution)

    boundaries /= 126.
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

    # Define outliers here after analyzing results
    flowers_outliers = np.zeros(N)
    # flowers_outliers[[40]] = 1

    conventional_outliers = np.zeros(N)
    if farm == 'small':
        conventional_outliers[[57,60]] = 1
    elif farm == 'medium':
        conventional_outliers[[13]] = 1
    elif farm == 'large': # 18 is a copy of 16
        conventional_outliers[[18,77]] = 1
    flowers_min_dist = np.zeros(N)
    conventional_min_dist = np.zeros(N)

    # Collect statistics
    initial_aep = np.zeros(N)
    flowers_optimal_aep = np.zeros(N)
    flowers_solver_time = np.zeros(N)
    flowers_iterations = np.zeros(N)
    flowers_exit_codes = {}
    conventional_optimal_aep = np.zeros(N)
    conventional_solver_time = np.zeros(N)
    conventional_iterations = np.zeros(N)
    conventional_exit_codes = {}

    for n in range(N):
        initial_aep[n] = solutions_flowers[n]['init_aep']
        flowers_optimal_aep[n] = solutions_flowers[n]['opt_aep']
        flowers_solver_time[n] = solutions_flowers[n]['total_time']
        flowers_iterations[n] = solutions_flowers[n]['iter']
        conventional_optimal_aep[n] = solutions_conventional[n]['opt_aep']
        conventional_solver_time[n] = solutions_conventional[n]['total_time']
        conventional_iterations[n] = solutions_conventional[n]['iter']

        exit_code = solutions_flowers[n]['exit_code']
        if exit_code in flowers_exit_codes:
            flowers_exit_codes[exit_code] += 1
        else:
            flowers_exit_codes[exit_code] = 1
        
        exit_code = solutions_conventional[n]['exit_code']
        if exit_code in conventional_exit_codes:
            conventional_exit_codes[exit_code] += 1
        else:
            conventional_exit_codes[exit_code] = 1
    
    # # Find best solutions (OPTION)
    # print(np.argmax(flowers_optimal_aep))
    # print(np.argmax(conventional_optimal_aep))

    if farm == "small":
        time_scale = 60
        time_scale_string = '[min]'
        xticks = [0,2,4,6,8,10]
        yticks = [0,2,4,6,8,10]
    elif farm == "medium":
        time_scale = 60
        time_scale_string = '[min]'
        xticks = [0,5,10,15,20,25,30,35]
        yticks = [0,5,10,15,20,25,30,35]
    elif farm == "large":
        time_scale = 3600
        time_scale_string = '[hr]'
        xticks = [0,25,50,75,100,125,150]
        yticks = [0,25,50,75,100,125,150]

    print("FLOWERS Median Cost: {:.1f} s".format(np.median(flowers_solver_time)))
    print("Conventional Median Cost: {:.1f} ".format(np.median(conventional_solver_time)/time_scale) + time_scale_string[1:-1])
    print("Speed-Up: {:.2f}x".format(np.median(conventional_solver_time)/np.median(flowers_solver_time)))
    print("FLOWERS Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(flowers_solver_time)/3600))
    print("Conventional Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(conventional_solver_time)/3600))
    print("Speed-Up: {:.2f}x".format(np.sum(conventional_solver_time)/np.sum(flowers_solver_time)))
    print("FLOWERS Std. Cost: {:.2f} s".format(np.std(flowers_solver_time)))
    print("Conventional Std. Cost: {:.2f} ".format(np.std(conventional_solver_time)/time_scale) + time_scale_string[1:-1])
    print("Improvement: {:.2f}%".format((np.std(flowers_solver_time) - np.std(conventional_solver_time))/np.std(conventional_solver_time)*100))
    print()
    print("FLOWERS Median AEP: {:.1f} GWh".format(np.median(flowers_optimal_aep)/1e9))
    print("Conventional Median AEP: {:.1f} GWh".format(np.median(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.median(flowers_optimal_aep) - np.median(conventional_optimal_aep))/np.median(conventional_optimal_aep)*100))
    print("FLOWERS Best AEP: {:.1f} GWh".format(np.max(flowers_optimal_aep)/1e9))
    print("Conventional Best AEP: {:.1f} GWh".format(np.max(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.max(flowers_optimal_aep) - np.max(conventional_optimal_aep))/np.max(conventional_optimal_aep)*100))
    print("FLOWERS Std. AEP: {:.2f} GWh".format(np.std(flowers_optimal_aep)/1e9))
    print("Conventional Std. AEP: {:.2f} GWh".format(np.std(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.std(flowers_optimal_aep) - np.std(conventional_optimal_aep))/np.std(conventional_optimal_aep)*100))
    print()
    print("FLOWERS Median Iterations: {:.0f}".format(np.median(flowers_iterations)))
    print("Conventional Median Iterations: {:.0f}".format(np.median(conventional_iterations)))
    print("Improvement: {:.2f}%".format((np.median(flowers_iterations) - np.median(conventional_iterations))/np.median(conventional_iterations)*100))
    print("FLOWERS Std. Iterations: {:.1f}".format(np.std(flowers_iterations)))
    print("Conventional Std. Iterations: {:.1f}".format(np.std(conventional_iterations)))
    print("Improvement: {:.2f}%".format((np.std(flowers_iterations) - np.std(conventional_iterations))/np.std(conventional_iterations)*100))

    # Exit codes
    fig, ax = plt.subplots(1,1,figsize=(11,3))
    ax.bar(range(len(flowers_exit_codes)), list(flowers_exit_codes.values()), align='center',color=color_flowers,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(flowers_exit_codes)), xticklabels=list(flowers_exit_codes.keys()),title='FLOWERS')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'flowers_codes.png', dpi=dpi)

    fig, ax = plt.subplots(1,1,figsize=(11,3))
    ax.bar(range(len(conventional_exit_codes)), list(conventional_exit_codes.values()), align='center',color=color_conventional,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(conventional_exit_codes)), xticklabels=list(conventional_exit_codes.keys()),title='Conventional')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'conventional_codes.png', dpi=dpi)

    # Superimposed layouts
    fig, ax = plt.subplots(1,3,figsize=(11,4.5))
    for n in range(N):
        solution = solutions_flowers[n]
        layout = []
        for i in range(len(solution['init_x'])):
            layout.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral, alpha=0.2)
        ax[0].add_collection(layouts)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='Initial',xticks=xticks,yticks=yticks)
    ax[0].grid(linestyle=':')

    for n in range(N):
        solution = solutions_flowers[n]
        layout = []

        # # Calculate minimum distance (OPTION)
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # flowers_min_dist[n] = np.min(dd)

        if flowers_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_flowers, alpha=0.2)
        ax[1].add_collection(layouts)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='FLOWERS-AD',xticks=xticks,yticks=yticks)
    ax[1].grid(linestyle=':')

    for n in range(N):
        solution = solutions_conventional[n]
        layout = []

        # # Calculate minimum distance (OPTION)
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # conventional_min_dist[n] = np.min(dd)

        if conventional_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_conventional, alpha=0.2)
        ax[2].add_collection(layouts)
    ax[2].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[2].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='Conventional-FD',xticks=xticks,yticks=yticks)
    ax[2].grid(linestyle=':')

    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'layouts.png', dpi=dpi)

    # # Print infeasible solutions (OPTION)
    # print(np.where(flowers_min_dist < 1))
    # print(np.where(conventional_min_dist < 1))

    # Best cases
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    if farm == "small":
        nn = 8
    elif farm == "medium":
        nn = 25
    elif farm == "large":
        nn = 75
    solution = solutions_flowers[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_flowers,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_flowers,zorder=2)
    # ax[0].add_collection(layout0)
    ax[0].add_collection(layout1)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS-AD'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    # ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS Best: {:.2f} GWh'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    ax[0].grid(linestyle=':')

    if farm == "small":
        nn = 2
    elif farm == "medium":
        nn = 56
    elif farm == "large":
        nn = 28
    solution = solutions_conventional[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_conventional,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_conventional,zorder=2)
    # ax[1].add_collection(layout0)
    ax[1].add_collection(layout1)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='Conventional-FD'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    # ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='Conventional Best: {:.2f} GWh'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    ax[1].grid(linestyle=':')
    # ax.legend(handles=[tmp0,tmp1],loc='upper right')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'best.png', dpi=dpi)

    # Mask outliers
    flowers_optimal_aep = np.ma.masked_where(flowers_outliers,flowers_optimal_aep)
    conventional_optimal_aep = np.ma.masked_where(conventional_outliers,conventional_optimal_aep)

    # AEP vs. Time
    fig, ax = plt.subplots(1,2,figsize=(11,5))
    ax[0].scatter(flowers_solver_time/time_scale,flowers_optimal_aep/1e9,20,alpha=0.75,color=color_flowers,label='FLOWERS-AD')
    ax[0].scatter(conventional_solver_time/time_scale,conventional_optimal_aep/1e9,20,alpha=0.75,color=color_conventional,label='Conventional-FD')
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax[0].vlines(np.median(flowers_solver_time)/time_scale,ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax[0].hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--')
    ax[0].vlines(np.median(conventional_solver_time)/time_scale,ylim[0],ylim[1],colors=color_conventional,linestyles='--')
    ax[0].set(xlabel='Solver Time ' + time_scale_string,ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
    ax[0].grid(linestyle=':')
    ax[0].set_axisbelow(True)
    # ax[0].legend(loc='lower right')

    # AEP vs. Iterations
    # fig, ax = plt.subplots(1,1,figsize=(11,5))
    ax[1].scatter(flowers_iterations,flowers_optimal_aep/1e9,20,alpha=0.75,color=color_flowers,label='FLOWERS-AD')
    ax[1].scatter(conventional_iterations,conventional_optimal_aep/1e9,20,alpha=0.75,color=color_conventional,label='Conventional-FD')
    xlim = ax[1].get_xlim()
    ylim = ax[1].get_ylim()
    ax[1].hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax[1].vlines(np.median(flowers_iterations),ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax[1].hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_conventional,linestyles='--')
    ax[1].vlines(np.median(conventional_iterations),ylim[0],ylim[1],colors=color_conventional,linestyles='--')
    ax[1].set(xlabel='Iterations',xlim=xlim,ylim=ylim,yticklabels=[])
    ax[1].grid(linestyle=':')
    ax[1].set_axisbelow(True)
    ax[1].legend(loc='lower right')

    fig.text(0.49,0.92,'(a)',fontweight='bold')
    fig.text(0.955,0.92,'(b)',fontweight='bold')

    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'qoi.png', dpi=dpi)

    # AEP Gain per Study
    fig, ax = plt.subplots(1,1,figsize=(11,5))
    off_val = 0.15
    mrk_sz = 2
    line_width = 1
    for n in range(N):
        if flowers_outliers[n] == 0:
            ax.scatter(n-off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n-off_val,flowers_optimal_aep[n]/1e9,mrk_sz,color=color_flowers)
            ax.vlines(n-off_val,initial_aep[n]/1e9,flowers_optimal_aep[n]/1e9,color=color_flowers,linewidth=line_width)

        if conventional_outliers[n] == 0:
            ax.scatter(n+off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n+off_val,conventional_optimal_aep[n]/1e9,mrk_sz,color=color_conventional)
            ax.vlines(n+off_val,initial_aep[n]/1e9,conventional_optimal_aep[n]/1e9,color=color_conventional,linewidth=line_width)
    ax.scatter([],[],20,color=color_neutral,label='Initial')
    ax.scatter([],[],20,color=color_flowers,label='FLOWERS-AD')
    ax.scatter([],[],20,color=color_conventional,label='Conventional-FD')
    ax.set(xlabel='Index',ylabel='AEP [GWh]')
    ax.legend()
    ax.grid(linestyle=':')
    fig.tight_layout()

    if save:
        plt.savefig(fig_name + 'aepgain.png', dpi=dpi)

    plt.show()

# ###########################################################################
# # FLOWERS-FD GRADIENTS
# ###########################################################################

if figs == "gradients":

    # Load files
    N = 100

    solutions_flowers = []
    solutions_conventional = []

    file_base = 'solutions/opt_' + farm
    fig_name = './figures/opt_gradients_' + farm + '_'

    for n in range(N):
        file_name = file_base + '_flowers_analytical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_flowers.append(solution)

        file_name = file_base + '_flowers_numerical_' + str(n) + '.p'
        solution, wr, boundaries = pickle.load(open(file_name,'rb'))
        solutions_conventional.append(solution)

    boundaries /= 126.
    boundaries = np.append(boundaries,np.reshape(boundaries[:,0],(2,1)),axis=1)

    # Define outliers here after analyzing results
    flowers_outliers = np.zeros(N)
    # flowers_outliers[[40]] = 1

    conventional_outliers = np.zeros(N)
    # if farm == 'large': # 18 is a copy of 16
    #     conventional_outliers[[18,77]] = 1
    flowers_min_dist = np.zeros(N)
    conventional_min_dist = np.zeros(N)

    # Collect statistics
    initial_aep = np.zeros(N)
    flowers_optimal_aep = np.zeros(N)
    flowers_solver_time = np.zeros(N)
    flowers_iterations = np.zeros(N)
    flowers_exit_codes = {}
    conventional_optimal_aep = np.zeros(N)
    conventional_solver_time = np.zeros(N)
    conventional_iterations = np.zeros(N)
    conventional_exit_codes = {}

    for n in range(N):
        initial_aep[n] = solutions_flowers[n]['init_aep']
        flowers_optimal_aep[n] = solutions_flowers[n]['opt_aep']
        flowers_solver_time[n] = solutions_flowers[n]['total_time']
        flowers_iterations[n] = solutions_flowers[n]['iter']
        conventional_optimal_aep[n] = solutions_conventional[n]['opt_aep']
        conventional_solver_time[n] = solutions_conventional[n]['total_time']
        conventional_iterations[n] = solutions_conventional[n]['iter']

        exit_code = solutions_flowers[n]['exit_code']
        if exit_code in flowers_exit_codes:
            flowers_exit_codes[exit_code] += 1
        else:
            flowers_exit_codes[exit_code] = 1
        
        exit_code = solutions_conventional[n]['exit_code']
        if exit_code in conventional_exit_codes:
            conventional_exit_codes[exit_code] += 1
        else:
            conventional_exit_codes[exit_code] = 1
    
    # # Find best solutions (OPTION)
    # print(np.argmax(flowers_optimal_aep))
    # print(np.argmax(conventional_optimal_aep))

    if farm == "small":
        time_scale = 1
        time_scale_string = '[s]'
        xticks = [0,2,4,6,8,10]
        yticks = [0,2,4,6,8,10]
    elif farm == "medium":
        time_scale = 1
        time_scale_string = '[s]'
        xticks = [0,5,10,15,20,25,30,35]
        yticks = [0,5,10,15,20,25,30,35]
    elif farm == "large":
        time_scale = 3600
        time_scale_string = '[hr]'
        xticks = [0,25,50,75,100,125,150]
        yticks = [0,25,50,75,100,125,150]

    print("FLOWERS-AD Median Cost: {:.1f} s".format(np.median(flowers_solver_time)))
    print("FLOWERS-FD Median Cost: {:.1f} ".format(np.median(conventional_solver_time)/time_scale) + time_scale_string[1:-1])
    print("Speed-Up: {:.2f}x".format(np.median(conventional_solver_time)/np.median(flowers_solver_time)))
    print("FLOWERS-AD Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(flowers_solver_time)/3600))
    print("FLOWERS-FD Aggregate Cost: {:.2f} cpu-hrs".format(np.sum(conventional_solver_time)/3600))
    print("Speed-Up: {:.2f}x".format(np.sum(conventional_solver_time)/np.sum(flowers_solver_time)))
    print("FLOWERS Std. Cost: {:.2f} s".format(np.std(flowers_solver_time)))
    print("Conventional Std. Cost: {:.2f} ".format(np.std(conventional_solver_time)/time_scale) + time_scale_string[1:-1])
    print("Improvement: {:.2f}%".format((np.std(flowers_solver_time) - np.std(conventional_solver_time))/np.std(conventional_solver_time)*100))
    print()
    print("FLOWERS-AD Median AEP: {:.1f} GWh".format(np.median(flowers_optimal_aep)/1e9))
    print("FLOWERS-FD Median AEP: {:.1f} GWh".format(np.median(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.median(flowers_optimal_aep) - np.median(conventional_optimal_aep))/np.median(conventional_optimal_aep)*100))
    print("FLOWERS-AD Best AEP: {:.1f} GWh".format(np.max(flowers_optimal_aep)/1e9))
    print("FLOWERS-FD Best AEP: {:.1f} GWh".format(np.max(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.max(flowers_optimal_aep) - np.max(conventional_optimal_aep))/np.max(conventional_optimal_aep)*100))
    print("FLOWERS-AD Std. AEP: {:.2f} GWh".format(np.std(flowers_optimal_aep)/1e9))
    print("FLOWERS-FD Std. AEP: {:.2f} GWh".format(np.std(conventional_optimal_aep)/1e9))
    print("Improvement: {:.2f}%".format((np.std(flowers_optimal_aep) - np.std(conventional_optimal_aep))/np.std(conventional_optimal_aep)*100))
    print()
    print("FLOWERS-AD Median Iterations: {:.0f}".format(np.median(flowers_iterations)))
    print("FLOWERS-FD Median Iterations: {:.0f}".format(np.median(conventional_iterations)))
    print("Improvement: {:.2f}%".format((np.median(flowers_iterations) - np.median(conventional_iterations))/np.median(conventional_iterations)*100))
    print("FLOWERS-AD Std. Iterations: {:.1f}".format(np.std(flowers_iterations)))
    print("FLOWERS-FD Std. Iterations: {:.1f}".format(np.std(conventional_iterations)))
    print("Improvement: {:.2f}%".format((np.std(flowers_iterations) - np.std(conventional_iterations))/np.std(conventional_iterations)*100))

    # Exit codes
    fig, ax = plt.subplots(1,1,figsize=(11,3))
    ax.bar(range(len(flowers_exit_codes)), list(flowers_exit_codes.values()), align='center',color=color_flowers,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(flowers_exit_codes)), xticklabels=list(flowers_exit_codes.keys()),title='FLOWERS-AD')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'flowers_codes.png', dpi=dpi)

    fig, ax = plt.subplots(1,1,figsize=(11,3))
    ax.bar(range(len(conventional_exit_codes)), list(conventional_exit_codes.values()), align='center',color=color_numerical,width=0.2)
    ax.set(xlabel='SNOPT Exit Codes', ylabel='Count',xticks=range(len(conventional_exit_codes)), xticklabels=list(conventional_exit_codes.keys()),title='FLOWERS-FD')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'conventional_codes.png', dpi=dpi)

    # Superimposed layouts
    fig, ax = plt.subplots(1,3,figsize=(11,4.5))
    for n in range(N):
        solution = solutions_flowers[n]
        layout = []
        for i in range(len(solution['init_x'])):
            layout.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layouts = coll.PatchCollection(layout, color=color_neutral, alpha=0.2)
        ax[0].add_collection(layouts)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='Initial',xticks=xticks,yticks=yticks)
    ax[0].grid(linestyle=':')

    for n in range(N):
        solution = solutions_flowers[n]
        layout = []

        # # Calculate minimum distance (OPTION)
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # flowers_min_dist[n] = np.min(dd)

        if flowers_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_flowers, alpha=0.2)
        ax[1].add_collection(layouts)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='FLOWERS-AD',xticks=xticks,yticks=yticks)
    ax[1].grid(linestyle=':')

    for n in range(N):
        solution = solutions_conventional[n]
        layout = []

        # # Calculate minimum distance (OPTION)
        # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
        # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
        # dd = np.sqrt(xx**2 + yy**2)
        # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
        # conventional_min_dist[n] = np.min(dd)

        if conventional_outliers[n] == 1:
            continue

        for i in range(len(solution['opt_x'])):
            layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layouts = coll.PatchCollection(layout, color=color_numerical, alpha=0.2)
        ax[2].add_collection(layouts)
    ax[2].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[2].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='FLOWERS-FD',xticks=xticks,yticks=yticks)
    ax[2].grid(linestyle=':')

    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'layouts.png', dpi=dpi)

    # # Print infeasible solutions (OPTION)
    # print(np.where(flowers_min_dist < 1))
    # print(np.where(conventional_min_dist < 1))

    # Best cases
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    if farm == "small":
        nn = 8
    elif farm == "medium":
        nn = 25
    elif farm == "large":
        nn = 75
    solution = solutions_flowers[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_flowers,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_flowers,zorder=2)
    # ax[0].add_collection(layout0)
    ax[0].add_collection(layout1)
    ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS-AD'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    # ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS-AD Best: {:.2f} GWh'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    ax[0].grid(linestyle=':')

    if farm == "small":
        nn = 8
    elif farm == "medium":
        nn = 25
    elif farm == "large":
        nn = 26
    solution = solutions_conventional[nn]
    layout_init = []
    layout_final = []
    for i in range(len(solution['opt_x'])):
        layout_init.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
        layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    tmp0 = plt.Circle(([],[]),1/2,color=color_neutral,label='Initial Layout')
    tmp1 = plt.Circle(([],[]),1/2,color=color_numerical,label='Optimal Layout')

    layout0 = coll.PatchCollection(layout_init, color=color_neutral)
    layout1 = coll.PatchCollection(layout_final, color=color_numerical,zorder=2)
    # ax[1].add_collection(layout0)
    ax[1].add_collection(layout1)
    ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS-FD'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    # ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title='FLOWERS-FD Best: {:.2f} GWh'.format(solution['opt_aep']/1e9),xticks=xticks,yticks=yticks)
    ax[1].grid(linestyle=':')
    # ax.legend(handles=[tmp0,tmp1],loc='upper right')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'best.png', dpi=dpi)

    # Mask outliers
    flowers_optimal_aep = np.ma.masked_where(flowers_outliers,flowers_optimal_aep)
    conventional_optimal_aep = np.ma.masked_where(conventional_outliers,conventional_optimal_aep)

    # AEP vs. Time
    fig, ax = plt.subplots(1,2,figsize=(11,5))
    ax[0].scatter(flowers_solver_time/time_scale,flowers_optimal_aep/1e9,20,alpha=0.75,color=color_flowers,label='FLOWERS-AD')
    ax[0].scatter(conventional_solver_time/time_scale,conventional_optimal_aep/1e9,20,alpha=0.75,color=color_numerical,label='FLOWERS-FD')
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax[0].vlines(np.median(flowers_solver_time)/time_scale,ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax[0].hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_numerical,linestyles='--')
    ax[0].vlines(np.median(conventional_solver_time)/time_scale,ylim[0],ylim[1],colors=color_numerical,linestyles='--')
    ax[0].set(xlabel='Solver Time ' + time_scale_string,ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
    ax[0].grid(linestyle=':')
    ax[0].set_axisbelow(True)
    # ax[0].legend(loc='lower right')

    # AEP vs. Iterations
    ax[1].scatter(flowers_iterations,flowers_optimal_aep/1e9,20,alpha=0.75,color=color_flowers,label='FLOWERS-AD')
    ax[1].scatter(conventional_iterations,conventional_optimal_aep/1e9,20,alpha=0.75,color=color_numerical,label='FLOWERS-FD')
    xlim = ax[1].get_xlim()
    ylim = ax[1].get_ylim()
    ax[1].hlines(np.median(flowers_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_flowers,linestyles='--')
    ax[1].vlines(np.median(flowers_iterations),ylim[0],ylim[1],colors=color_flowers,linestyles='--')
    ax[1].hlines(np.median(conventional_optimal_aep)/1e9,xlim[0],xlim[1],colors=color_numerical,linestyles='--')
    ax[1].vlines(np.median(conventional_iterations),ylim[0],ylim[1],colors=color_numerical,linestyles='--')
    ax[1].set(xlabel='Iterations',xlim=xlim,ylim=ylim,yticklabels=[])
    ax[1].grid(linestyle=':')
    ax[1].set_axisbelow(True)
    ax[1].legend(loc='lower right')

    fig.text(0.49,0.92,'(a)',fontweight='bold')
    fig.text(0.955,0.92,'(b)',fontweight='bold')
    fig.tight_layout()
    if save:
        plt.savefig(fig_name + 'qoi.png', dpi=dpi)

    # AEP Gain per Study
    fig, ax = plt.subplots(1,1,figsize=(11,5))
    off_val = 0.15
    mrk_sz = 2
    line_width = 1
    for n in range(N):
        if flowers_outliers[n] == 0:
            ax.scatter(n-off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n-off_val,flowers_optimal_aep[n]/1e9,mrk_sz,color=color_flowers)
            ax.vlines(n-off_val,initial_aep[n]/1e9,flowers_optimal_aep[n]/1e9,color=color_flowers,linewidth=line_width)

        if conventional_outliers[n] == 0:
            ax.scatter(n+off_val,initial_aep[n]/1e9,mrk_sz,color=color_neutral, zorder=100)
            ax.scatter(n+off_val,conventional_optimal_aep[n]/1e9,mrk_sz,color=color_numerical)
            ax.vlines(n+off_val,initial_aep[n]/1e9,conventional_optimal_aep[n]/1e9,color=color_numerical,linewidth=line_width)
    ax.scatter([],[],20,color=color_neutral,label='Initial')
    ax.scatter([],[],20,color=color_flowers,label='FLOWERS-AD')
    ax.scatter([],[],20,color=color_numerical,label='FLOWERS-FD')
    ax.set(xlabel='Index',ylabel='AEP [GWh]')
    ax.legend()
    fig.tight_layout()

    if save:
        plt.savefig(fig_name + 'aepgain.png', dpi=dpi)

    plt.show()

###########################################################################
# FLOWERS PARAMETER SENSITIVITY
###########################################################################

if figs == "parameter":
    def scale_lightness(rgb, scale_l):
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

    # Load files
    N = 100

    solutions_parameter = [list() for _ in range(10)]

    cc = co.ColorConverter.to_rgb(color_flowers)
    color_parameter = [scale_lightness(cc, scale) for scale in [1.6,1.45,1.3,1.15,1,0.85,0.7,0.55,0.4,0.25]]
    cmap = co.LinearSegmentedColormap.from_list('colors',color_parameter,N=10)

    # cmap = cm.get_cmap('GnBu')
    # color_parameter = [cmap(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]

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
    parameter_solver_time = np.zeros((10,N))
    parameter_iterations = np.zeros((10,N))

    for n in range(N):
        initial_aep[n] = solutions_parameter[0][n]['init_aep']
        for k in range(10):
            parameter_optimal_aep[k,n] = solutions_parameter[k][n]['opt_aep']
            parameter_solver_time[k,n] = solutions_parameter[k][n]['total_time']
            parameter_iterations[k,n] = solutions_parameter[k][n]['iter']
    
    # Find best solutions (OPTION)
    # print(np.argmax(parameter_optimal_aep,axis=1))

    if farm == "small":
        time_scale = 1
        time_scale_string = '[s]'
        xticks = [0,2,4,6,8,10]
        yticks = [0,2,4,6,8,10]
    elif farm == "medium":
        time_scale = 1
        time_scale_string = '[s]'
        xticks = [0,5,10,15,20,25,30,35]
        yticks = [0,5,10,15,20,25,30,35]
    elif farm == "large":
        time_scale = 1
        time_scale_string = '[s]'
        xticks = [0,25,50,75,100,125,150]
        yticks = [0,25,50,75,100,125,150]

    # # Superimposed layouts
    # fig, ax = plt.subplots(1,3,figsize=(11,4.5))
    # for n in range(N):
    #     solution = solutions_flowers[n]
    #     layout = []
    #     for i in range(len(solution['init_x'])):
    #         layout.append(plt.Circle((solution['init_x'][i]/126., solution['init_y'][i]/126.), 1/2))
    #     layouts = coll.PatchCollection(layout, color=color_neutral, alpha=0.2)
    #     ax[0].add_collection(layouts)
    # ax[0].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    # ax[0].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='Initial Layouts')
    # ax[0].grid(True)

    # for n in range(N):
    #     solution = solutions_flowers[n]
    #     layout = []

    #     # # Calculate minimum distance (OPTION)
    #     # xx = (solution['opt_x'] - np.reshape(solution['opt_x'],(-1,1)))/126.
    #     # yy = (solution['opt_y'] - np.reshape(solution['opt_y'],(-1,1)))/126.
    #     # dd = np.sqrt(xx**2 + yy**2)
    #     # dd = np.ma.masked_where(np.eye(np.shape(xx)[0]),dd)
    #     # flowers_min_dist[n] = np.min(dd)

    #     if flowers_outliers[n] == 1:
    #         continue

    #     for i in range(len(solution['opt_x'])):
    #         layout.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

    #     layouts = coll.PatchCollection(layout, color=color_flowers, alpha=0.2)
    #     ax[1].add_collection(layouts)
    # ax[1].plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    # ax[1].set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal', title='FLOWERS Layouts')
    # ax[1].grid(True)

    # fig.tight_layout()
    # if save:
    #     plt.savefig(fig_name + 'layouts.png', dpi=dpi)

    # # Print infeasible solutions (OPTION)
    # print(np.where(flowers_min_dist < 1))
    # print(np.where(conventional_min_dist < 1))

    # Best cases
    fig, ax = plt.subplots(1,1,figsize=(11,4.5))
    if farm == "small":
        nn = [20,80,50,24,8,55,7,55,95,83]
        case_title = 'Best: 251.8 $\pm$ 0.5 GWh'
    elif farm == "medium":
        nn = [33,25,85,1,25,58,58,32,25,25]
        # case_title = 'Best: 1037.9 $\pm$ 0.4 GWh'
        case_title = ''
    elif farm == "large":
        nn = [16,49,49,1,75,1,1,1,1,1]
        case_title = 'Best: 4905.0 $\pm$ 116 GWh'
    
    # print(np.mean(np.max(parameter_optimal_aep,axis=1))/1e9)
    # print((np.max(np.max(parameter_optimal_aep,axis=1)) - np.mean(np.max(parameter_optimal_aep,axis=1)))/1e9)
    # print((np.mean(np.max(parameter_optimal_aep,axis=1)) - np.min(np.max(parameter_optimal_aep,axis=1)))/1e9)

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

    for k in range(10):
        solution = solutions_parameter[k][nn[k]]
        layout_final = []
        for i in range(len(solution['opt_x'])):
            layout_final.append(plt.Circle((solution['opt_x'][i]/126., solution['opt_y'][i]/126.), 1/2))

        layout = coll.PatchCollection(layout_final, color=color_parameter[k], alpha=0.5, zorder=2)
        ax.add_collection(layout)

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
    # das

    ax.plot(boundaries[0],boundaries[1],color='k',linewidth=2,zorder=1)
    ax.set(xlabel='$x/D$', ylabel='$y/D$', aspect='equal',title=case_title,xticks=xticks,yticks=yticks)
    ax.grid(linestyle=':')

    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=co.Normalize(vmin=0.01,vmax=0.11)),ax=ax,label='$k$', ticks=[0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095,0.105],fraction=0.07,shrink=0.85)
    cbar.set_ticklabels(['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.10'])
    fig.tight_layout(rect=[0.2,0,0.75,1])
    if save:
        plt.savefig(fig_name + 'layouts.png', dpi=dpi)

    # # Mask outliers
    # flowers_optimal_aep = np.ma.masked_where(flowers_outliers,flowers_optimal_aep)
    # conventional_optimal_aep = np.ma.masked_where(conventional_outliers,conventional_optimal_aep)

    # AEP vs. Time
    fig, ax = plt.subplots(1,2,figsize=(11,5))
    for k in range(10):
        ax[0].scatter(parameter_solver_time[k]/time_scale,parameter_optimal_aep[k]/1e9,20,alpha=0.75,color=color_parameter[k])
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    for k in range(10):
        ax[0].hlines(np.median(parameter_optimal_aep[k])/1e9,xlim[0],xlim[1],colors=color_parameter[k],linestyles='--')
        ax[0].vlines(np.median(parameter_solver_time[k])/time_scale,ylim[0],ylim[1],colors=color_parameter[k],linestyles='--')
    ax[0].set(xlabel='Solver Time ' + time_scale_string,ylabel='Optimal AEP [GWh]',xlim=xlim,ylim=ylim)
    ax[0].grid(linestyle=':')
    ax[0].set_axisbelow(True)

    # AEP vs. Iterations
    for k in range(10):
        ax[1].scatter(parameter_iterations[k],parameter_optimal_aep[k]/1e9,20,alpha=0.75,color=color_parameter[k])
    xlim = ax[1].get_xlim()
    ylim = ax[1].get_ylim()
    for k in range(10):
        ax[1].hlines(np.median(parameter_optimal_aep[k])/1e9,xlim[0],xlim[1],colors=color_parameter[k],linestyles='--')
        ax[1].vlines(np.median(parameter_iterations[k]),ylim[0],ylim[1],colors=color_parameter[k],linestyles='--')
    ax[1].set(xlabel='Iterations',xlim=xlim,ylim=ylim,yticklabels=[])
    ax[1].grid(linestyle=':')
    ax[1].set_axisbelow(True)
    fig.tight_layout()
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=co.Normalize(vmin=0.01,vmax=0.11)),ax=ax,label='$k$', ticks=[0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095,0.105],fraction=0.07,pad=0.04,shrink=0.85)
    cbar.set_ticklabels(['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.10'])
    fig.text(0.44,0.92,'(a)',fontweight='bold')
    fig.text(0.855,0.92,'(b)',fontweight='bold')
    if save:
        plt.savefig(fig_name + 'qoi.png', dpi=dpi)

    plt.show()