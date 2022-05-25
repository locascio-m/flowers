# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import pickle

import floris.tools.optimization.pyoptsparse as opt

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
boundaries = [(0.0, 0.0), (0.0, 2000.0), (1600.0, 1600.0), (2000.0, 0.0), (0.0, 0.0)]
# boundaries = [(0.0, 0.0), (0.0, 1000.0), (800.0, 800.0), (1000.0, 0.0), (0.0, 0.0)]

# Wind farm layout
layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=25)

# Initialize optimization comparison
geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=7, wd_resolution=5.0)

# FLOWERS optimization
model = layout.LayoutOptimization(fi, geo.boundaries)
tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
sol = tmp.optimize()
geo.save_flowers_solution(model,sol)

# FLORIS optimization
# model = opt.layout.Layout(fli, boundaries, freq=geo.freq_floris)
# tmp = opt.optimization.Optimization(model=model, solver='SLSQP')
# sol = tmp.optimize()
# geo.save_floris_solution(model,sol)

# Output results
# geo.compare_optimization(stats=True)
# geo.plot_optimal_layouts()
# vis.plot_wind_rose(wind_rose)

geo.show_flowers_solution(stats=True)
geo.plot_flowers_layout()

# Save results
# pickle.dump(geo, open('solutions/sol.p','wb'))

plt.show()