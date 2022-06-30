# FLOWERS

# Michael LoCascio

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
n_turb = 31

# Wind farm boundaries
boundaries = [
    (2714.4, 4049.4),
    (2132.7, 938.8),
    (2092.8, 591.6),
    (2078.9, 317.3),
    (2076.1, 148.5),
    (2076.6, 0.0),
    (2076.5, 6.5),
    (1208.6, 847.0),
    (0.0, 2017.7),
    (1496.7, 4027.2),
    (1531.8, 4006.2),
    (1931.2, 3818.5),
    (2058.3, 3783.6),
    (2192.8, 3792.9),
    (2316.8, 3846.4),
    (2416.0, 3939.1),
    (2528.6, 4089.0),
    (2550.9, 4126.3)
]

# Wind rose resolution
num_terms = 37
wd_resolution = 5.0

# Output file name
file_name = 'solutions/multi' + str(sys.argv[1]) + '.p'
hist_file = 'output/hist' + str(sys.argv[1]) + '.hist'

### Optimization study

# Randomize wind farm layout
layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=n_turb)

# Initialize optimization interface
geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=num_terms, wd_resolution=wd_resolution)

# FLORIS optimization
print("Solving FLORIS optimization.")
fli.calculate_wake()
prob = LayoutOptimizationPyOptSparse(fli, geo.boundaries, freq=geo.freq_floris, solver='SLSQP', storeHistory=hist_file)
sol = prob.optimize()
geo.save_floris_solution(sol, history=hist_file)

# FLOWERS optimization
print("Solving FLOWERS optimization.")
model = layout.LayoutOptimization(fi, geo.boundaries)
tmp = opt.optimization.Optimization(model=model, solver='SLSQP', storeHistory=hist_file)
sol = tmp.optimize()
geo.save_flowers_solution(sol, history=hist_file)

# Save results
pickle.dump(geo, open(file_name,'wb'))