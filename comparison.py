# FLOWERS

# Michael LoCascio

import numpy as np
import model as set
import pickle
import tools as tl
import warnings

"""
This file compares the AEP between FLOWERS and FLORIS for three different
wind roses and layouts. 

"""

warnings.filterwarnings("ignore")

# Overall parameters
D = 126.0
num_terms = -1
wd_resolution = 1.0
ws_avg = True
case = 0

# Run sweep
var = []
aep_flowers = []
aep_floris = []
time_flowers = []
time_floris = []

if case == 0:
    N_max = 50
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(12)
    TI = 0.06
    for idx in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,28,32,36,40,44,48,52,56,60]: #
        layout_x = np.linspace(0., (idx-1)*7*D, idx)
        layout_y = np.zeros(idx)
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False, iter=1)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 1:
    N_max = 12
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(7)
    TI = 0.12
    for idx in np.arange(1,N_max+1):
        xx = np.linspace(0., (idx-1)*7*D, idx)
        layout_x, layout_y = np.meshgrid(xx,xx)
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx**2)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 2:
    N_max = 25
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(13)
    TI = 0.07
    for idx in np.arange(3,N_max+1):
        xx = np.linspace(0., 2.*idx*D+0.001, 3)
        layout_x, layout_y = np.meshgrid(xx,xx)
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 3:
    N_max = 20
    file = 'solutions/bench' + str(case) + '.p'
    boundaries = [(0.,0.),(7*7*D,0.),(7*7*D,7*7*D),(0.,7*7*D)]       
    wind_rose = tl.load_wind_rose(7)
    TI = 0.06

    for idx in range(N_max):
        layout_x, layout_y = tl.random_layout(boundaries, n_turb=50, D=D, min_dist=3.0)

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen')
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 4 or case == 5:
    file = 'solutions/bench' + str(case) + '.p'  
    if case == 5:   
        xx = np.linspace(0., 7*7*D, 8)
        layout_x, layout_y = np.meshgrid(xx,xx)
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()
    else:
        layout_x = np.array([0.])
        layout_y = np.array([0.])
    rose_index = [1,2,3,4,5,6,7,8,9]
    TI = 0.07
    
    for idx in rose_index:
        wind_rose = tl.load_wind_rose(idx)
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss', TI=TI)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 6:
    file = 'solutions/bench' + str(case) + '.p'     
    xx = np.linspace(0., 7*7*D, 8)
    layout_x, layout_y = np.meshgrid(xx,xx)
    layout_x = layout_x.flatten()
    layout_y = layout_y.flatten()
    wind_rose = tl.load_wind_rose(0)
    TI = 0.11
    WD_list = [360,180,120,90,72,45,40,36,30,24,20,18,15,12,10,9,8,5,4]
    res_list = 360 / np.array(WD_list)
    N_list = [182,160,140,120,100,80,60,50,40,30,25,20,18,16,14,12,10,9,8,7,6,5,4,3,2]
    
    for WD in res_list:
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=2, wd_resolution=WD, ws_avg=ws_avg, display=False)

        aep_floris.append(aep[1])
        time_floris.append(time[1])
    
    for N in N_list:
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=N, wd_resolution=90.0, ws_avg=ws_avg, display=False)

        aep_flowers.append(aep[0])
        time_flowers.append(time[0])
    
    var = (N_list, WD_list)

if case == 7:
    N_max = 25
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(12)
    TI = 0.07
    for idx in np.arange(3,N_max+1):
        layout_x  = np.array([0., idx*D, 2*idx*D])
        layout_y = np.array([0., 0., 0.])

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 8:
    N_max = 25
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(12)
    TI = 0.07
    for idx in np.arange(3,N_max+1):
        layout_x  = np.array([0., idx*D])
        layout_y = np.array([0., 0.])

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 9:
    N = 10
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(12)
    TI = 0.06
    for idx in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]: #,4,5,6,7,8,9,10,12,14,16,18,20,22,24,28,32,36,40,44,48,52,56,60
        layout_x = np.array([0.0, 7.*D])
        layout_x = np.tile(layout_x,idx)
        layout_y = np.linspace(0., (idx-1)*7.*D, idx)
        layout_y = np.repeat(layout_y,2)
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

if case == 10:
    N = 10
    file = 'solutions/bench' + str(case) + '.p'
    wind_rose = tl.load_wind_rose(12)
    TI = 0.06
    for idx in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]: #1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40
        layout_x = np.linspace(0., (idx-1)*7.*D, idx)
        layout_x = np.repeat(layout_x,2)
        layout_y = np.array([0.0, 7.*D])
        layout_y = np.tile(layout_y,idx)
        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='jensen', z0=1e-3)
        aep, time = geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg, display=False)

        var.append(idx)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])

pickle.dump((var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose, TI), open(file,'wb'))