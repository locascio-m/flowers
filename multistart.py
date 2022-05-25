# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import numpy as np
import pickle

import floris.tools.optimization.pyoptsparse as opt

import flowers_interface as flow
import layout
import tools as tl
import visualization as vis

"""
This file runs a specified number of randomized layout optimization studies
with FLOWERS for a given wind rose and wind plant boundary. The wind plant
layout is randomized with a set number of turbines.

The ModelComparison objects are saved to 'sol#.p' files for post-processing
in comparison.py
"""

# Inputs

# Number of random starts
multi = 20

# Wind rose (sampled from stored wind roses)
wind_rose = tl.load_wind_rose(1)

# Wind farm boundaries
boundaries = [(0.0, 0.0), (0.0, 2000.0), (1600.0, 1600.0), (2000.0, 0.0), (0.0, 0.0)]

# Multistart study
for i in range(multi):

    print("CASE {:.0f}: Initialized.".format(i))

    # Randomize wind farm layout
    layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=16)

    # Initialize optimization interface
    geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
    fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=5, wd_resolution=5.0)

    # FLOWERS optimization
    model = layout.LayoutOptimization(fi, geo.boundaries)
    tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
    sol = tmp.optimize()
    geo.save_flowers_solution(model,sol)

    # Save results
    file_name = 'solutions/sol' + str(i+100) + '.p'
    pickle.dump(geo, open(file_name,'wb'))

    print("CASE {:.0f}: Solved.".format(i))

print("All cases completed!")