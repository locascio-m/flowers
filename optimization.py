# FLOWERS

# Michael LoCascio

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
# boundaries = [
#     (2714.4, 4049.4),
#     (2132.7, 938.8),
#     (2092.8, 591.6),
#     (2078.9, 317.3),
#     (2076.1, 148.5),
#     (2076.6, 0.0),
#     (2076.5, 6.5),
#     (1208.6, 847.0),
#     (0.0, 2017.7),
#     (1496.7, 4027.2),
#     (1531.8, 4006.2),
#     (1931.2, 3818.5),
#     (2058.3, 3783.6),
#     (2192.8, 3792.9),
#     (2316.8, 3846.4),
#     (2416.0, 3939.1),
#     (2528.6, 4089.0),
#     (2550.9, 4126.3)
# ]

# Wind farm layout
layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=3)

file_name = 'multi_test.p'
hist_file = 'hist_test.hist'
summary_file1 = 'snopt_floris_summary.out'
summary_file2 = 'snopt_flowers_summary.out'
print_file1 = 'snopt_floris_print.out'
print_file2 = 'snopt_flowers_print.out'

# Initialize optimization comparison
geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=37, wd_resolution=5.0)

# FLORIS optimization TODO: why is calculate_wake() not working?
fli.calculate_wake()
prob = LayoutOptimizationPyOptSparse(fli, geo.boundaries, freq=geo.freq_floris, solver='SNOPT', storeHistory=hist_file, optOptions={'iPrint': -1, 'Print file': print_file1, 'Summary file': summary_file1})
sol = prob.optimize()
geo.save_floris_solution(sol, history=hist_file)

# FLOWERS optimization
model = layout.LayoutOptimization(fi, geo.boundaries)
tmp = opt.optimization.Optimization(model=model, solver='SNOPT', storeHistory=hist_file, optOptions={'iPrint': -2, 'Print file': print_file2, 'Summary file': summary_file2})
sol = tmp.optimize()
geo.save_flowers_solution(sol, history=hist_file)

# Output results
geo.show_optimization_comparison(stats=True)
geo.plot_optimal_layouts()
vis.plot_wind_rose(wind_rose)
# geo.plot_optimization_histories(flowers_mov="flowers.mp4", floris_mov="floris.mp4")

# Save results
# pickle.dump(geo, open('solutions/test.p','wb'))

plt.show()