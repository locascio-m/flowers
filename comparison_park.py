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
case = 2

# Run sweep
var = []
aep_flowers = []
aep_floris = []
time_flowers = []
time_floris = []

# Resolution over wind speeds and wind directions
if case == 0:
    file = 'solutions/park' + str(case) + '.p'     
    xx = np.linspace(0., 7*7*D, 8)
    layout_x, layout_y = np.meshgrid(xx,xx)
    layout_x = layout_x.flatten()
    layout_y = layout_y.flatten()
    wr_list = [1,2,6,9]
    WD_list = [360,180,120,90,72,45,40,36,30,24,20,18,15,12,10,9,8,5,4]
    wd_res_list = 360 / np.array(WD_list)
    WS_list = [26,24,22,20,18,16,14,13,12,11,10,9,8,7,6,5,4,3]
    ws_res_list = 26. / np.array(WS_list)
    N_list = [180,160,140,120,100,80,60,50,40,30,25,20,18,16,14,12,10,9,8,7,6,5,4,3,2]

    aep_floris = np.zeros((len(wd_res_list),len(ws_res_list)+1,len(wr_list)))
    time_floris = np.zeros((len(wd_res_list),len(ws_res_list)+1,len(wr_list)))
    aep_flowers = np.zeros((len(N_list),len(wr_list)))
    time_flowers = np.zeros((len(N_list),len(wr_list)))
    for i in range(len(wr_list)):
        wind_rose = tl.load_wind_rose(wr_list[i])
        for d in range(len(wd_res_list)):
            for s in range(len(ws_res_list)):
                geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
                aep, time = geo.compare_aep(num_terms=2, wd_resolution=wd_res_list[d], ws_resolution=ws_res_list[s], display=False)
                aep_floris[d,s,i]=aep[1]
                time_floris[d,s,i]=time[1]
            
            geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
            aep, time = geo.compare_aep(num_terms=2, wd_resolution=wd_res_list[d], ws_avg=True, display=False)
            aep_floris[d,-1,i]=aep[1]
            time_floris[d,-1,i]=time[1]
        
        for n in range(len(N_list)):
            geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
            aep, time = geo.compare_aep(num_terms=N_list[n], wd_resolution=90.0, ws_avg=True, display=False)

            aep_flowers[n,i] = aep[0]
            time_flowers[n,i] = time[0]
    
    var = (N_list, WD_list, WS_list)

# Average cost as a function of number of turbines
if case == 1:
    N_max = 12
    file = 'solutions/park' + str(case) + '.p'
    wr_list = [1,2,3,4,5,6,7,8,9]
    time_flowers = np.zeros((N_max,len(wr_list)))
    time_floris = np.zeros((N_max,len(wr_list)))
    for idx in np.arange(1,N_max+1):
        print('{:.0f} Array'.format(idx))
        xx = np.linspace(0., (idx-1)*7*D, idx)
        layout_x, layout_y = np.meshgrid(xx,xx)
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()

        for wr_idx in range(len(wr_list)):
            wind_rose = tl.load_wind_rose(wr_list[wr_idx])
            geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
            aep, tmp = geo.compare_aep(num_terms=-1, wd_resolution=1.0, ws_avg=True, display=False)
            time_flowers[idx-1,wr_idx] = tmp[0]
            time_floris[idx-1,wr_idx] = tmp[1]

        var.append(idx**2)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])

# Randomized number of turbines, layout, and wind rose
if case == 2:
    N_max = 200
    file = 'solutions/park' + str(case) + '.p'    
    N_turb = np.random.randint(2,101,N_max)
    N_wr = np.random.randint(1,10,N_max)

    aep_flowers_full = []
    aep_flowers_fast = []
    aep_park_full = []
    aep_park_fast = []
    time_flowers_full = []
    time_flowers_fast = []
    time_park_full = []
    time_park_fast = []

    for idx in range(N_max):
        print(idx)
        
        wind_rose = tl.load_wind_rose(N_wr[idx])
        layout_x, layout_y = tl.discrete_layout(n_turb=N_turb[idx], D=D, min_dist=3.0)

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')

        aep, time = geo.compare_aep(num_terms=-1, wd_resolution=1.0, ws_avg=False, display=False)
        aep_flowers_full.append(aep[0])
        aep_park_full.append(aep[1])
        time_flowers_full.append(time[0])
        time_park_full.append(time[1])

        aep, time = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, display=False)
        aep_flowers_fast.append(aep[0])
        aep_park_fast.append(aep[1])
        time_flowers_fast.append(time[0])
        time_park_fast.append(time[1])

    var = (N_wr, N_turb)

    pickle.dump((
        var, 
        aep_flowers_full, 
        aep_flowers_fast, 
        aep_park_full, 
        aep_park_fast, 
        time_flowers_full, 
        time_flowers_fast, 
        time_park_full, 
        time_park_fast
        ), 
        open(file,'wb'))

# Randomized number of turbines, layout, and wind rose (low resolution)
if case == 3:
    N_max = 200
    file = 'solutions/park' + str(case) + '.p'    
    N_turb = np.random.randint(2,101,N_max)
    N_wr = np.random.randint(1,10,N_max)
    spacing = []

    for idx in range(N_max):
        wind_rose = tl.load_wind_rose(N_wr[idx])
        layout_x, layout_y, ss = tl.discrete_layout(n_turb=N_turb[idx], D=D, min_dist=3.0, spacing=True)

        geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
        aep, time = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, display=False, iter=5)

        spacing.append(ss)
        aep_flowers.append(aep[0])
        aep_floris.append(aep[1])
        time_flowers.append(time[0])
        time_floris.append(time[1])
    var = (spacing, N_wr, N_turb)

# Error as a function of number of turbines
if case == 4:
    N_max = 2
    file = 'solutions/park' + str(case) + '.p'
    wr_list = [1,2,3,4,5,6,7,8,9]
    aep_flowers = np.zeros((N_max,len(wr_list)))
    aep_floris = np.zeros((N_max,len(wr_list)))
    for idx in np.arange(1,N_max+1):
        print('{:.0f} Array'.format(idx))
        xx = np.linspace(0., (idx-1)*7*D, idx)
        layout_x, layout_y = np.meshgrid(xx,xx)
        layout_x = layout_x.flatten()
        layout_y = layout_y.flatten()

        for wr_idx in range(len(wr_list)):
            wind_rose = tl.load_wind_rose(wr_list[wr_idx])
            geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='park')
            aep, tmp = geo.compare_aep(num_terms=5, wd_resolution=5.0, ws_avg=True, display=False)
            aep_flowers[idx-1,wr_idx] = aep[0]
            aep_floris[idx-1,wr_idx] = aep[1]

        var.append(idx**2)
        time_flowers.append(tmp[0])
        time_floris.append(tmp[1])

# pickle.dump((var, aep_flowers, aep_floris, time_flowers, time_floris, layout_x, layout_y, wind_rose), open(file,'wb'))