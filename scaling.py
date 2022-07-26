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

Usage:
    $ python multistart.py <index> <opt_type> <scaling>

    *index* : index for file names and seed for randomized initial layout.
    *opt_type* : specify 'flowers' for FLOWERS only, 'floris' for FLORIS only,
        or 'both' for FLOWERS and FLORIS optimization.
"""

if __name__ == "__main__":

    ### Inputs

    # Wind rose (sampled from stored wind roses)
    wind_rose = tl.load_wind_rose(1)

    # Number of turbines
    n_turb = 1

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
        file_name = 'solutions/multi_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.p'
        hist_file = 'output/hist_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
    elif str(sys.argv[2]) == "both":
        flowers_flag = True
        floris_flag = True
        file_name = 'solutions/multi_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.p'
        hist_file = 'output/hist_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
    elif str(sys.argv[2]) == "flowers":
        flowers_flag = True
        file_name = 'solutions/flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.p'
        hist_file = 'output/hist_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.hist'
        summary_flowers_name = 'output/snopt_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_flowers_name = 'output/print_flowers_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
    elif str(sys.argv[2]) == "floris":
        floris_flag = True
        file_name = 'solutions/floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.p'
        hist_file = 'output/hist_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.hist'
        summary_floris_name = 'output/snopt_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'
        print_floris_name = 'output/print_floris_' + str(sys.argv[1]) + '_scale_' + str(sys.argv[3]) + '.out'

    # Design variable scaling
    scaleDV = float(sys.argv[3])
    scaleCON = 1.0

    ### Optimization study

    # Randomize wind farm layout
    layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=n_turb, idx=int(sys.argv[1]))

    # Initialize optimization interface
    geo = flow.ModelComparison(wind_rose, layout_x, layout_y)
    fi, fli = geo.initialize_optimization(boundaries=boundaries, num_terms=num_terms, wd_resolution=wd_resolution)

    # FLORIS optimization
    if floris_flag:
        print("Solving FLORIS optimization.")
        fli.calculate_wake()
        prob = LayoutOptimizationPyOptSparse(
            fli,
            geo.boundaries,
            freq=geo.freq_floris,
            scale_dv=scaleDV,
            scale_con=scaleCON,
            solver='SNOPT',
            storeHistory=hist_file,
            optOptions={'iPrint': -1, 'Print file': print_floris_name, 'Summary file': summary_floris_name},
            timeLimit=86400,
        )
        sol = prob.optimize()
        geo.save_floris_solution(sol, history=hist_file)

    # FLOWERS optimization
    if flowers_flag:
        print("Solving FLOWERS optimization.")
        model = layout.LayoutOptimization(fi, geo.boundaries, scale_dv=scaleDV, scale_con=scaleCON)
        tmp = opt.optimization.Optimization(
            model=model, 
            solver='SNOPT', 
            storeHistory=hist_file,
            optOptions={'iPrint': -2, 'Print file': print_flowers_name, 'Summary file': summary_flowers_name},
            timeLimit=86400,
        )
        sol = tmp.optimize()
        geo.save_flowers_solution(sol, history=hist_file)

    # Save results
    pickle.dump(geo, open(file_name,'wb'))