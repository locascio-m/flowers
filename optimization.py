# FLOWERS

# Michael LoCascio

import numpy as np
import matplotlib.pyplot as plt
import pickle

import floris.tools.optimization.pyoptsparse as opt
from floris.tools.optimization.layout_optimization.layout_optimization_pyoptsparse import LayoutOptimizationPyOptSparse

import flowers_interface as flow
import layout
import tools as tl
import visualization as vis

"""
This file runs two layout optimization studies---one for FLORIS and
one for FLOWERS---based on a specified wind rose and wind plant layout.

The wind rose, wind plant boundaries, and wind plant layout are initialized.
Then, the comparison is set up and the optimization is run for FLOWERS and FLORIS.
The results are saved in the ModelComparison() object for post-processing.
"""

# Inputs

# Wind rose (sampled from stored wind roses)
wind_rose = tl.load_wind_rose(1)

# Wind farm boundaries
boundaries = [(0.0, 0.0), (0.0, 800.0), (800.0, 800.0), (800.0, 0.0), (0.0, 0.0)]

# Wind farm layout
layout_x = np.array([1.,270., 799., 1., 400., 799., 1., 240., 799.])
layout_y = np.array([1., 1., 1., 200., 260., 400., 700., 750., 799.])
# layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=10)

# Initialize optimization comparison
geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=7, wd_resolution=5.0)

# FLORIS optimization
# model = opt.layout.Layout(fli, boundaries, freq=geo.freq_floris)
# tmp = opt.optimization.Optimization(model=model, solver='SLSQP')

# TODO: why is calculate_wake() not working?
fli.calculate_wake()
prob = LayoutOptimizationPyOptSparse(fli, geo.boundaries, freq=geo.freq_floris, solver='SLSQP')
prob.optimize()
geo.save_floris_solution(prob)

# FLOWERS optimization
model = layout.LayoutOptimization(fi, geo.boundaries)
tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
sol = tmp.optimize()
geo.save_flowers_solution(model,sol)

# Output results

# FLOWERS and FLORIS
geo.compare_optimization(stats=True)
geo.plot_optimal_layouts()
vis.plot_wind_rose(wind_rose)

# FLOWERS only
# geo.show_flowers_solution(stats=True)
# geo.plot_flowers_layout()

# FLORIS only
# geo.show_floris_solution(stats=True)
# geo.plot_floris_layout()

# Save results
# pickle.dump(geo, open('solutions/sol.p','wb'))

plt.show()