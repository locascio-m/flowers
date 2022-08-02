# FLOWERS

# Michael LoCascio

import pickle
import sys

from floris.tools.optimization.layout_optimization.layout_optimization_pyoptsparse import LayoutOptimizationPyOptSparse
import floris.tools.optimization.pyoptsparse as opt

import model as set
import tools as tl

"""
This file runs one parallelized instance of a randomized layout optimization study
with FLOWERS and FLORIS for a given wind rose and wind plant boundary. The wind plant
layout is randomized with a set number of turbines.

The ModelComparison objects are saved to 'multi#.p' files for post-processing
in comparison.py

Usage:
    $ python multistart.py <index> <opt_type>

    *index* : index for file names and seed for randomized initial layout.
    *opt_type* : specify 'flowers' for FLOWERS only, 'floris' for FLORIS only,
        or 'both' for FLOWERS and FLORIS optimization.
"""

if __name__ == "__main__":

    ### Inputs

    # Wake model
    wake = "gauss"

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
    
    ### Output definitions

    # Specify which optimizations to run
    flowers_flag = False
    floris_flag = False

    if len(sys.argv) == 2:
        flowers_flag = True
        floris_flag = True
        file_name = 'solutions/multi_' + str(sys.argv[1]) + '.p'
        hist_file = 'output/hist_' + str(sys.argv[1]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '.out'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '.out'
    elif str(sys.argv[2]) == "both":
        flowers_flag = True
        floris_flag = True
        file_name = 'solutions/multi_' + str(sys.argv[1]) + '.p'
        hist_file = 'output/hist_' + str(sys.argv[1]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '.out'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '.out'
    elif str(sys.argv[2]) == "flowers":
        flowers_flag = True
        file_name = 'solutions/flowers_' + str(sys.argv[1]) + '.p'
        hist_file = 'output/hist_flowers_' + str(sys.argv[1]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '.out'
    elif str(sys.argv[2]) == "floris":
        floris_flag = True
        file_name = 'solutions/floris_' + str(sys.argv[1]) + '.p'
        hist_file = 'output/hist_floris_' + str(sys.argv[1]) + '.hist'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '.out'

    ### Optimization study

    # Randomize wind farm layout
    layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=n_turb, idx=int(sys.argv[1]))

    # Initialize optimization interface
    geo = set.ModelComparison(wind_rose, layout_x, layout_y, model=wake)
    geo.initialize_optimization(boundaries, num_terms=num_terms, wd_resolution=wd_resolution)

    # FLORIS optimization
    if floris_flag:
        geo.run_floris_optimization(
            history_file=hist_file,
            output_file=summary_floris_name
            )

    # FLOWERS optimization
    if flowers_flag:
        geo.run_flowers_optimization(
            history_file=hist_file,
            output_file=summary_flowers_name
            )

    # Save results
    pickle.dump(geo, open(file_name,'wb'))