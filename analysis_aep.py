import numpy as np
import pickle
from shapely.geometry import Polygon, Point
import warnings

import model_interface as inter
import tools as tl

warnings.filterwarnings("ignore")

"""
This file conducts three different numerical experiments to compare
AEP estimates and their sensitivity between the FLOWERS and Conventional
AEP models

"""

case = 3


###########################################################################
# RANDOMIZED LAYOUT / WIND ROSE
###########################################################################
if case == 0:
    # Analysis options
    N_max = 100
    num_terms= 10
    wd_resolution = 5.0
    ws_avg = True

    # Initialization
    file = 'solutions/aep' + str(case) + '.p' 
    wr_list = [] 
    N_turb = np.random.randint(2,101,N_max)
    N_wr = np.random.randint(1,10,N_max)

    aep_flowers = np.zeros(N_max)
    aep_floris = np.zeros(N_max)
    time_flowers = np.zeros(N_max)
    time_floris = np.zeros(N_max)

    # Gather wind roses
    for i in np.arange(1,10):
        wr_list.append(tl.load_wind_rose(i))

    model = inter.AEPInterface(wr_list[0], np.zeros(2), np.zeros(2))

    # Calculate AEP for random layout and wind rose
    for idx in range(N_max):
        print('Case ' + str(idx+1) + ' of ' + str(N_max))
        wind_rose = wr_list[N_wr[idx]-1]
        layout_x, layout_y = tl.discrete_layout(n_turb=N_turb[idx], min_dist=3.0)

        model.reinitialize(wind_rose=wind_rose, layout_x=layout_x, layout_y=layout_y, num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg)
        aep, time = model.compare_aep(display=False)

        aep_flowers[idx] = aep[0]
        aep_floris[idx] = aep[1]

        time_flowers[idx] = time[0]
        time_floris[idx] = time[1]

    var = (N_wr, N_turb)

    pickle.dump((var, aep_flowers, aep_floris, time_flowers, time_floris), open(file,'wb'))

###########################################################################
# RANDOMIZED RESOLUTION
###########################################################################
if case == 1:
    # Analysis options
    N_max = 100
    N_turb = 50
    ws_avg = True

    # Initialization
    file = 'solutions/aep' + str(case) + '.p' 
    wr_list = [] 
    N_wr = np.random.randint(1,10,N_max)
    N_terms = np.random.randint(2,180,N_max)
    N_bins = np.random.randint(4,360,N_max)

    aep_flowers = np.zeros(N_max)
    aep_floris = np.zeros(N_max)
    time_flowers = np.zeros(N_max)
    time_floris = np.zeros(N_max)

    # Gather wind roses
    for i in np.arange(1,10):
        wr_list.append(tl.load_wind_rose(i))

    model = inter.AEPInterface(wr_list[0], np.zeros(2), np.zeros(2))

    # Calculate AEP for random layout, wind rose, and resolution
    for idx in range(N_max):
        print('Case ' + str(idx+1) + ' of ' + str(N_max))
        wind_rose = wr_list[N_wr[idx]-1]
        layout_x, layout_y = tl.discrete_layout(n_turb=N_turb, min_dist=3.0)

        model.reinitialize(wind_rose=wind_rose, layout_x=layout_x, layout_y=layout_y, num_terms=N_terms[idx], wd_resolution=360/N_bins[idx], ws_avg=ws_avg)
        aep, time = model.compare_aep(display=False)

        aep_flowers[idx] = aep[0]
        aep_floris[idx] = aep[1]

        time_flowers[idx] = time[0]
        time_floris[idx] = time[1]

    var = (N_terms, N_bins)

    pickle.dump((var, aep_flowers, aep_floris, time_flowers, time_floris), open(file,'wb'))

###########################################################################
# LAYOUT MUTATION
###########################################################################
if case == 2:
    # Analysis options
    N = 25
    step = 126./5
    np.random.seed(4)
    flowers_terms = [100,20,10]
    conv_resolution = [1,5,10]
    ws_avg = True

    # Define parameters
    X = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
    Y = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])
    wr = tl.load_wind_rose(7)
    nt = len(X)
    file = 'solutions/aep' + str(case) + '.p' 
    resolution = [flowers_terms, conv_resolution]

    # Store AEP
    aep_flowers = np.ones((3,N+1))
    aep_park = np.ones((3,N+1))
    aep_gauss = np.ones((3,N+1))
    x_all = np.zeros((N+1,nt))
    y_all = np.zeros((N+1,nt))

    # Define FLOWERS and Park models
    model_high = inter.AEPInterface(wr, X, Y, conventional_model='park')
    model_high.reinitialize(num_terms=flowers_terms[0], wd_resolution=conv_resolution[0], ws_avg=ws_avg)

    model_med = inter.AEPInterface(wr, X, Y, conventional_model='park')
    model_med.reinitialize(num_terms=flowers_terms[1], wd_resolution=conv_resolution[1], ws_avg=ws_avg)

    model_low = inter.AEPInterface(wr, X, Y, conventional_model='park')
    model_low.reinitialize(num_terms=flowers_terms[2], wd_resolution=conv_resolution[2], ws_avg=ws_avg)

    # Define Gauss models
    gauss_high = inter.AEPInterface(wr, X, Y, conventional_model='gauss')
    gauss_high.reinitialize(num_terms=flowers_terms[0], wd_resolution=conv_resolution[0], ws_avg=ws_avg)

    gauss_med = inter.AEPInterface(wr, X, Y, conventional_model='gauss')
    gauss_med.reinitialize(num_terms=flowers_terms[1], wd_resolution=conv_resolution[1], ws_avg=ws_avg)

    gauss_low = inter.AEPInterface(wr, X, Y, conventional_model='gauss')
    gauss_low.reinitialize(num_terms=flowers_terms[2], wd_resolution=conv_resolution[2], ws_avg=ws_avg)

    # Store initial data
    aep_flowers[0,0] = model_high.compute_flowers_aep()
    aep_flowers[1,0] = model_med.compute_flowers_aep()
    aep_flowers[2,0] = model_low.compute_flowers_aep()

    aep_park[0,0] = model_high.compute_floris_aep()
    aep_park[1,0] = model_med.compute_floris_aep()
    aep_park[2,0] = model_low.compute_floris_aep()

    aep_gauss[0,0] = gauss_high.compute_floris_aep()
    aep_gauss[1,0] = gauss_med.compute_floris_aep()
    aep_gauss[2,0] = gauss_low.compute_floris_aep()

    x_all[0] = X
    y_all[0] = Y

    for i in np.arange(1,N+1):
        print('Case ' + str(i) + ' of ' + str(N))

        # Add random Gaussian noise to turbine positions
        X += np.random.normal(0.,step,nt)
        Y += np.random.normal(0.,step,nt)

        model_high.reinitialize(layout_x=X, layout_y=Y)
        model_med.reinitialize(layout_x=X, layout_y=Y)
        model_low.reinitialize(layout_x=X, layout_y=Y)
        gauss_high.reinitialize(layout_x=X, layout_y=Y)
        gauss_med.reinitialize(layout_x=X, layout_y=Y)
        gauss_low.reinitialize(layout_x=X, layout_y=Y)

        # Compute new AEP for each model
        aep_flowers[0,i] = model_high.compute_flowers_aep()
        aep_flowers[1,i] = model_med.compute_flowers_aep()
        aep_flowers[2,i] = model_low.compute_flowers_aep()

        aep_park[0,i] = model_high.compute_floris_aep()
        aep_park[1,i] = model_med.compute_floris_aep()
        aep_park[2,i] = model_low.compute_floris_aep()

        aep_gauss[0,i] = gauss_high.compute_floris_aep()
        aep_gauss[1,i] = gauss_med.compute_floris_aep()
        aep_gauss[2,i] = gauss_low.compute_floris_aep()

        x_all[i] = np.copy(X)
        y_all[i] = np.copy(Y)

    # Normalize AEP by initial value
    aep_flowers /= np.expand_dims(aep_flowers[:,0],-1)
    aep_park /= np.expand_dims(aep_park[:,0],-1)
    aep_gauss /= np.expand_dims(aep_gauss[:,0],-1)

    pickle.dump((x_all, y_all, aep_flowers, aep_park, aep_gauss, resolution, wr), open(file,'wb'))

###########################################################################
# SOLUTION SPACE SMOOTHNESS
###########################################################################
if case == 3:
    # Resolution options
    flowers_terms = [180,90,60,45,36,30,25,20,15,10,5]
    conv_resolution = [1,2,3,4,5,6,8,10,12,15,18]

    # Load layout, boundaries, and rose
    X0 = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
    Y0 = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])
    idx = 4
    boundaries = [(0., 0.),(0, 14*126.),(14*126, 14*126.),(14*126, 0.)]
    nx = 16*4+1
    ny = 16*4+1

    wind_rose = tl.load_wind_rose(7)

    # Initialization
    file = 'solutions/aep' + str(case) + '.p'
    xmin = np.min(X0) - 126.
    xmax = np.max(X0) + 126.
    ymin = np.min(Y0) - 126.
    ymax = np.max(Y0) + 126.

    model = inter.AEPInterface(wind_rose, X0, Y0, conventional_model='park')
    gauss = inter.AEPInterface(wind_rose, X0, Y0, num_terms=3, conventional_model='gauss')

    xx = np.linspace(xmin,xmax,nx,endpoint=True)
    yy = np.linspace(ymin,ymax,ny,endpoint=True)
    xx,yy = np.meshgrid(xx,yy)

    aep_flowers = np.zeros((len(flowers_terms),nx,ny))
    aep_park = np.zeros((len(conv_resolution),nx,ny))
    aep_gauss = np.zeros((len(conv_resolution),nx,ny))

    poly = Polygon(boundaries)

    for n in range(len(flowers_terms)):
        print('Resolution ' + str(n+1) + ' of ' + str(len(flowers_terms)))
        infeasible = np.zeros_like(xx)

        # Set resolution for models
        model.reinitialize(num_terms=flowers_terms[n],wd_resolution=conv_resolution[n],ws_avg=True)
        gauss.reinitialize(wd_resolution=conv_resolution[n],ws_avg=True)

        X = np.copy(X0)
        Y = np.copy(Y0)

        # Move select turbine around domain and compute AEP
        for i in range(ny):
            print('y ' + str(i+1) + ' of ' + str(ny))
            for j in range(nx):
                X[idx] = xx[i,j] + 1e-8
                Y[idx] = yy[i,j] + 1e-8
                model.reinitialize(layout_x=X,layout_y=Y)
                gauss.reinitialize(layout_x=X,layout_y=Y)

                aep = model.compare_aep(timer=False, display=False)
                aep_flowers[n,i,j] = aep[0]
                aep_park[n,i,j] = aep[1]

                aep_gauss[n,i,j] = gauss.compute_floris_aep(timer=False)

                pt = Point(xx[i,j], yy[i,j])

                # TODO: double-check Gauss NaN issue
                # if np.isnan(aep_gauss[n,i,j]):
                #     das

                # Mask points outside of boundary or colliding with other turbines
                if np.any(np.sqrt((X[idx] - np.delete(X,idx))**2 + (Y[idx] - np.delete(Y,idx))**2) <= 126.) or not pt.within(poly):
                    infeasible[i,j] = 1

    # Mask and normalize by highest AEP value
    infeasible = np.swapaxes(np.tile(np.expand_dims(infeasible, axis=2),len(flowers_terms)),0,2)
    aep_flowers = np.ma.masked_where(infeasible,aep_flowers)
    aep_park = np.ma.masked_where(infeasible,aep_park)
    aep_gauss = np.ma.masked_where(infeasible,aep_gauss)

    aep_flowers /= np.expand_dims(np.amax(aep_flowers,(1,2)),(1,2))
    aep_park /= np.expand_dims(np.amax(aep_park,(1,2)),(1,2))
    aep_gauss /= np.expand_dims(np.amax(aep_gauss,(1,2)),(1,2))
    
    xx /= 126.
    yy /= 126.

    pickle.dump((xx, yy, aep_flowers, aep_park, aep_gauss, flowers_terms, conv_resolution, wind_rose), open(file,'wb'))
