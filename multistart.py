# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from floris.tools.optimization.layout_optimization.layout_optimization_pyoptsparse import LayoutOptimizationPyOptSparse
import floris.tools.optimization.pyoptsparse as opt

import flowers_interface as flow
import layout
import tools as tl

"""
This file runs one parallelized instance of a randomized layout optimization study
with FLOWERS and FLORIS for a given wind rose and wind plant boundary. The wind plant
layout is randomized with a set number of turbines.

The ModelComparison objects are saved to 'multi#.p' files for post-processing
in comparison.py
"""

### Inputs

# Wind rose (sampled from stored wind roses)
wind_rose = tl.load_wind_rose(1)

# Number of turbines
n_turb = 2

# Wind farm boundaries
boundaries = [(0.0, 0.0), (0.0, 2000.0), (1600.0, 1600.0), (2000.0, 0.0), (0.0, 0.0)]

# Wind rose resolution
num_terms = 7
wd_resolution = 5.0

# Output file name
file_name = 'solutions/multi' + str(sys.argv[1]) + '.p'

### Optimization study

# Randomize wind farm layout
print("Generating wind farm layout.")
layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=n_turb)

# Initialize optimization interface
print("Initializing optimization problem.")
geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=num_terms, wd_resolution=wd_resolution)

# FLORIS optimization
print("Solving FLORIS optimization.")
fli.calculate_wake()
prob = LayoutOptimizationPyOptSparse(fli, geo.boundaries, freq=geo.freq_floris, solver='SLSQP')
sol = prob.optimize()
print("Saving FLORIS solution.")
geo.save_floris_solution(sol)

# FLOWERS optimization
print("Solving FLOWERS optimization.")
model = layout.LayoutOptimization(fi, geo.boundaries)
tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
sol = tmp.optimize()
print("Saving FLOWERS solution.")
geo.save_flowers_solution(sol)

# Save results
pickle.dump(geo, open(file_name,'wb'))


# if parallel:


#     for i in range(multi):

#         print("CASE {:.0f}: Initialized.".format(i))

#         # Randomize wind farm layout
#         layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=n_turb)

#         # Initialize optimization interface
#         geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
#         fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=7, wd_resolution=5.0)

#         # FLORIS optimization
#         fli.calculate_wake()
#         prob = LayoutOptimizationPyOptSparse(fli, geo.boundaries, freq=geo.freq_floris, solver='SLSQP')
#         sol = prob.optimize()
#         geo.save_floris_solution(sol)

#         # FLOWERS optimization
#         model = layout.LayoutOptimization(fi, geo.boundaries)
#         tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
#         sol = tmp.optimize()
#         geo.save_flowers_solution(sol)

#         # Save results
#         file_name = 'solutions/sol' + str(i+100) + '.p'
#         pickle.dump(geo, open(file_name,'wb'))

#         print("CASE {:.0f}: Solved.".format(i))

#     print("All cases completed!")