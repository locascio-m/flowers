import numpy as np
import model as set
import tools as tl
import matplotlib.pyplot as plt
import pickle
import visualization as vis
import warnings

save = False

xlabels = [
    'Number of Turbines',
    'Number of Turbines',
    'Turbine Spacing [D]',
    'Index',
    'Index',
    'Index',
    'None',
    'TI [%]',
    'TI [%]',
    'k',
    'Separation [D]'
    ]
titles = [
    'Case 0: Adding Turbines (Line), 7D Spacing',
    'Case 1: Adding Turbines in Grid, 7D Spacing',
    'Case 2: Spacing, 3x3 Grid',
    'Case 3: Random, 50 Turbines',
    'Case 4: Wind Roses, Free Turbine',
    'Case 5: Wind Roses, 8x8 Grid',
    'Case 6: Resolution, 8x8 Grid',
    'Case 7: TI, Two Turbines',
    'Case 8: TI, 8x8 Grid',
    'Case 9: k, Two Turbines',
    'Case 10: Spacing, Two Turbines'
]

for case in [0,1,2,3,4,5,6,7,8,9,10]:

    file_name = 'solutions/bench' + str(case) + '.p'

    if case == 6:
        var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose, TI = pickle.load(open(file_name,'rb'))

        aep_flowers = np.array(aep_flowers) / aep_flowers[0]
        aep_floris = np.array(aep_floris) / aep_floris[0]

        time_flowers = np.array(time_flowers) / time_flowers[0]
        time_floris = np.array(time_floris) / time_floris[0]

        num_terms = var[0]
        num_bins = var[1]

        # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2,figsize = (11,7))

        fig = plt.figure(figsize=(14,7))
        ax0 = fig.add_subplot(231)
        ax1 = fig.add_subplot(232, sharey=ax0)
        ax2 = fig.add_subplot(234, sharex=ax0)
        ax3 = fig.add_subplot(235, sharex=ax1, sharey=ax2)
        ax4 = fig.add_subplot(233, polar=True)
        ax5 = fig.add_subplot(236)

        ax0.plot(num_terms, time_flowers,'-o',markersize=3)
        ax0.set(xlabel='Number of Fourier Terms', ylabel='Normalized Time', title='FLOWERS')

        ax1.plot(num_bins, time_floris,'-o',markersize=3)
        ax1.set(xlabel='Number of Wind Directions', ylabel='Normalized Time', title='Conventional')

        ax2.plot(num_terms, aep_flowers,'-o',markersize=3)
        ax2.set(xlabel='Number of Fourier Terms', ylabel='Normalized AEP')

        ax3.plot(num_bins, aep_floris,'-o',markersize=3)
        ax3.set(xlabel='Number of Wind Directions', ylabel='Normalized AEP')

        fig.suptitle(titles[case])

    else:
        
        var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose, TI = pickle.load(open(file_name,'rb'))

        # if case == 4 or case == 5:
        #     tmp = [str(elem) for elem in var]
        #     var = range(len(tmp))

        aep_error = [(aep_flowers[i] - aep_floris[i]) / aep_floris[i] * 100 for i in range(len(aep_flowers))]
        time_factor = [time_floris[i] / time_flowers[i] for i in range(len(aep_flowers))]

        fig = plt.figure(figsize=(14,7))
        ax0 = fig.add_subplot(231)
        ax1 = fig.add_subplot(232)
        ax2 = fig.add_subplot(234)
        ax3 = fig.add_subplot(235)
        ax4 = fig.add_subplot(233, polar=True)
        ax5 = fig.add_subplot(236)

        ax0.plot(var,[elem / 1e9 for elem in aep_flowers],'-o',markersize=3)
        ax0.plot(var,[elem / 1e9 for elem in aep_floris],'-o',markersize=3)
        ax0.set(xlabel=xlabels[case], ylabel='AEP [GWh]')
        # if case == 4 or case == 5:
        #     ax0.set(xticklabels=tmp)
        ax0.legend(['FLOWERS','Conventional'])

        ax1.plot(var,time_flowers,'-o',markersize=3)
        ax1.plot(var,time_floris,'-o',markersize=3)
        ax1.set(xlabel=xlabels[case], ylabel='Time [s]')
        # if case == 4 or case == 5:
        #     ax1.set(xticklabels=tmp)

        ax2.plot(var, aep_error, 'g-o', markersize=3)
        ax2.set(xlabel=xlabels[case], ylabel='AEP Difference [%]')
        # if case == 4 or case == 5:
        #     ax2.set(xticklabels=tmp)

        ax3.plot(var, time_factor, 'g-o', markersize=3)
        ax3.set(xlabel=xlabels[case], ylabel='Time Factor [x]')
        # if case == 4 or case == 5:
        #     ax3.set(xticklabels=tmp)
        fig.suptitle(titles[case])

    vis.plot_wind_rose(wind_rose, ax=ax4)
    vis.plot_layout(layout_x, layout_y, ax=ax5)
    ax5.set(title='TI = {:.2f}'.format(TI))
    fig.tight_layout()
    if save:
        plt.savefig("/Users/locascio/Library/Mobile Documents/com~apple~CloudDocs/Research/FLOWERS Improvements/case_calibration" + str(case), dpi=500)

plt.show()