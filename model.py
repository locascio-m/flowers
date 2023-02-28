# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import numpy as np
import os
from pyoptsparse.pyOpt_history import History
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, LineString
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, LineString

import floris.tools as wfct
from floris.tools.optimization.layout_optimization.layout_optimization_pyoptsparse import LayoutOptimizationPyOptSparse

import flowers_interface as fi
import layout as lyt
import tools as tl
import visualization as vis

class ModelComparison:
    """
    ModelComparison is an interface for comparing AEP and optimization
    performance between FLOWERS and FLORIS. It streamlines the initialization
    of FLOWERS and FLORIS interfaces and post-processing operations.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        model (str, optional): FLORIS wake model choice from the following options:
            - 'jensen' (default): classical Jensen / top-hat model
            - 'gauss': Gauss model

    """

    def __init__(self, wind_rose, layout_x, layout_y, model="jensen", z0=1e-3):

        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.model = model

        # Internal variable for verifying optimization initialization and completion
        self.opt_init = False
        self.opt_floris = False
        self.opt_flowers = False

        # Initialize FLORIS from input file
        if model == "jensen":
            input_file = "./input/jensen.yaml"
        elif model == "gauss":
            input_file = "./input/gauss.yaml"
        self.floris = wfct.floris_interface.FlorisInterface(input_file)
        TI = 1 / np.log(90/z0)

        self.floris.reinitialize(
            layout=(layout_x.flatten(),layout_y.flatten()), 
            wind_shear=0,
            turbulence_intensity=TI)
        
        # Initialize wind direction-speed frequency array for AEP
        wd_array = np.array(self.wind_rose["wd"].unique(), dtype=float)
        ws_array = np.array(self.wind_rose["ws"].unique(), dtype=float)
        wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
        freq_interp = NearestNDInterpolator(self.wind_rose[["wd", "ws"]], self.wind_rose["freq_val"])
        freq = freq_interp(wd_grid, ws_grid)
        self.freq_floris = freq / np.sum(freq)
        self.bins_floris = len(wd_array)

        self.floris.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array)
        self.floris.calculate_wake()

        # Initialize FLORIS interface for post-processing with Gauss model
        self.post = wfct.floris_interface.FlorisInterface("./input/gauss.yaml")
        self.post.reinitialize(
            layout=(layout_x.flatten(),layout_y.flatten()), 
            wind_shear=0,
            turbulence_intensity=TI)
        self.post_freq = self.freq_floris

        self.post.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array)
        self.post.calculate_wake()

        # Import other parameters
        # TODO: import wake expansion rate from FLORIS
        self.diameter = self.floris.floris.farm.rotor_diameters[0][0][0]

        # Calibrate FLOWERS parameter
        N0 = 100

        # x_length = np.max(layout_x) - np.min(layout_x)
        # y_length = np.max(layout_y) - np.min(layout_y)
        # A = x_length*y_length
        # delta = len(layout_x) * np.pi * self.diameter**2 / (4 * A)
        delta = 0.8/49

        k0 = 1.7 * 0.41 / (np.log(90/z0))
        kInf = np.sqrt(k0**2 + 1.7 * 0.47 / (5 + delta**(-1/2))**2)
        # kInf = 3*k0

        # k = k0 * (1 + (kInf/k0 - 1) * np.tanh((len(layout_x) - 2) / (N0/2)))

        k = 0.05

        # if TI <= 0.09:
        #     k0 = 0.05
        # else:
        #     k0 = 0.075
        # k = k0 * (1 + 2 * np.tanh((len(layout_x) - 2) / 50))

        # Initialize FLOWERS interface
        self.flowers = fi.Flowers(
            self.wind_rose,
            self.layout_x,
            self.layout_y,
            k=k,
            D=self.diameter
            )


    ###########################################################################
    # Interface tools
    ###########################################################################

    def reinitialize_flowers(self):
        """
        Recompute FLOWERS Fourier coefficients (full discrete series).
        Returns the FLOWERS interface.

        """

        self.flowers.fourier_coefficients()
        self.terms_flowers = len(self.flowers.fs.a_wake)

        return self.flowers
    
    def truncate_flowers(self, num_terms):
        """
        Recompute FLOWERS Fourier coefficients with limited number of modes.
        Returns the FLOWERS interface.

        """

        self.flowers.fourier_coefficients(num_terms=num_terms)
        self.terms_flowers = num_terms

        return self.flowers
    
    def reinitialize_floris(self):
        """
        Reinitialize FLORIS interface with original wind rose.
        Returns the FLORIS interface.

        """

        wd_array = np.array(self.wind_rose["wd"].unique(), dtype=float)
        ws_array = np.array(self.wind_rose["ws"].unique(), dtype=float)  
        wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
        freq_interp = NearestNDInterpolator(self.wind_rose[["wd", "ws"]], self.wind_rose["freq_val"])
        freq = freq_interp(wd_grid, ws_grid)
        self.freq_floris = freq / np.sum(freq)

        self.floris.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array)
        
        return self.floris

    def resample_floris(self, ws_avg=False, wd_resolution=1.0):
        """
        Reinitialize FLORIS interface with resampled wind rose.
        Returns the FLORIS interface.

        Args:
            ws_avg (bool, optional): Indicate whether wind speed should be
                averaged for each wind direction bin.
            wd_resolution (float, optional): width of each resampled wind
                direction bin [deg]

        """

        # Resample wind rose with specified wind direction bins
        wr = self.wind_rose.copy(deep=True)

        if wd_resolution > 1.0:
            wr = tl.resample_wind_direction(
                wr, 
                wd=np.arange(0, 360, wd_resolution)
                )

        # Resample wind rose by average wind speed per wind direction
        if ws_avg:
            wr = tl.resample_average_ws_by_wd(wr)
            freq = wr.freq_val.to_numpy()
            self.freq_floris = freq / np.sum(freq)
            self.bins_floris = len(self.freq_floris)

            self.floris.reinitialize(
                wind_directions=wr.wd,
                wind_speeds=wr.ws,
                time_series=True,
                )

        else:
            wd_array = np.array(self.wind_rose["wd"].unique(), dtype=float)
            ws_array = np.array(self.wind_rose["ws"].unique(), dtype=float)  
            wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
            freq_interp = NearestNDInterpolator(self.wind_rose[["wd", "ws"]], self.wind_rose["freq_val"])
            freq = freq_interp(wd_grid, ws_grid)
            self.freq_floris = freq / np.sum(freq)
            self.bins_floris = len(wd_array)

            self.floris.reinitialize(
                wind_directions=wd_array,
                wind_speeds=ws_array)
        
        return self.floris

    def get_flowers_interface(self):
        """Get current instantiation of FLOWERS interface"""
        return self.flowers

    def get_floris_interface(self):
        """Get current instantiation of FLORIS interface"""
        return self.floris


    ###########################################################################
    # AEP methods
    ###########################################################################   

    def compute_flowers_aep(self, num_terms=None, timer=False):
        """
        Compute farm AEP using the FLOWERS model.

        Args:
            num_terms (int, optional): the number of Fourier modes to compute AEP
                in the range [1, ceiling(num_wind_directions/2)]
            timer (bool, optional): indicate whether wall timer should be
                used for calculation.

        Returns:
            aep (float): farm AEP [Wh]
            elapsed (float, optional): wall time of AEP calculation [s]

        """

        # Initialize the FLOWERS interface
        if num_terms == None:
            self.reinitialize_flowers()
        else:
            self.truncate_flowers(num_terms=num_terms)

        # Time AEP calculation
        if timer:
            t = time.time()
            aep = self.flowers.calculate_aep()
            elapsed = time.time() - t
            return aep, elapsed

        else:
            aep = self.flowers.calculate_aep()
            return aep
    
    def compute_floris_aep(self, ws_avg=False, wd_resolution=1.0, timer=False):
        """
        Compute farm AEP using the FLORIS model.

        Args:
            ws_avg (bool, optional): Indicate whether wind speed should be
                averaged for each wind direction bin.
            wd_resolution (float, optional): the width of the discrete wind
                direction bins to compute AEP
            timer (bool, optional): indicate whether wall timer should be
                used for calculation.

        Returns:
            aep (float): farm AEP [Wh]
            elapsed (float, optional): wall time of AEP calculation [s]

        """

        # Reinitialze the FLORIS interface
        self.reinitialize_floris()

        # Resample the FLORIS interface
        if wd_resolution > 1.0 or ws_avg:
            self.resample_floris(ws_avg=ws_avg, wd_resolution=wd_resolution)

        # Time AEP calculation
        if timer:
            t = time.time()
            self.floris.calculate_wake()

        # Reshape frequency array (if required) and compute AEP
            if ws_avg:
                self.freq_floris = np.expand_dims(self.freq_floris,1)
                aep = np.sum(self.floris.get_farm_power() * self.freq_floris * 8760)

            else:
                aep = self.floris.get_farm_AEP(freq=self.freq_floris)
            elapsed = time.time() - t

            return aep, elapsed

        else:
            self.floris.calculate_wake()
            if ws_avg:
                self.freq_floris = np.expand_dims(self.freq_floris,1)
                aep = np.sum(self.floris.get_farm_power() * self.freq_floris * 8760)

            else:
                aep = self.floris.get_farm_AEP(freq=self.freq_floris)
            aep = self.floris.get_farm_AEP(freq=self.freq_floris)
            return aep

    def compare_aep(self, iter=5, num_terms=None, ws_avg=False, wd_resolution=1.0, display=True):
        """
        Compute farm AEP using both models and compare. The calculation is
            repeated an optional number of instances to average computation time.
            A table of relevant information is printed to the terminal.

        Args:
            iter (int, optional): the number of times AEP should be computed
                to average the wall time of each calculation.
            num_terms (int, optional): for FLOWERS, the number of Fourier modes
                to compute AEP in the range [1, ceiling(num_wind_directions/2)]
            ws_avg (bool, optional): for FLORIS, to indicate whether wind speed 
                should be averaged for each wind direction bin.
            wd_resolution (float, optional): for FLORIS, the width of the discrete 
                wind direction bins to compute AEP

        """

        print("Comparing AEP between FLOWERS and FLORIS-" + self.model.capitalize() + ".")

        # Initialize containers
        aep_flowers = np.zeros(iter)
        aep_floris = np.zeros(iter)
        time_flowers = np.zeros(iter)
        time_floris = np.zeros(iter)

        # Compute AEP and average across iterations
        for n in range(iter):
            tmp = self.compute_flowers_aep(num_terms=num_terms, timer=True)
            aep_flowers[n] = tmp[0]
            time_flowers[n] = tmp[1]

            tmp = self.compute_floris_aep(wd_resolution=wd_resolution, timer=True, ws_avg=ws_avg)
            aep_floris[n] = tmp[0]
            time_floris[n] = tmp[1]
        
        aep_flowers_final = np.mean(aep_flowers)
        time_flowers_final = np.mean(time_flowers)
        aep_floris_final = np.mean(aep_floris)
        time_floris_final = np.mean(time_floris)
        if display:
            print("============================")
            print('    AEP Results    ')
            print('    Number of Turbines: {:.0f}'.format(len(self.layout_x)))
            print('    FLOWERS Terms: {:.0f}'.format(num_terms))
            print('    FLORIS Bins:   {:.0f}'.format(int(360/wd_resolution)))
            print('    FLORIS Average WS: ' + str(ws_avg))
            print("----------------------------")
            print("FLORIS  AEP:      {:.3f} GWh".format(aep_floris_final / 1.0e9))
            print("FLOWERS AEP:      {:.3f} GWh".format(aep_flowers_final / 1.0e9))
            print("Percent Difference:  {:.1f}%".format((aep_flowers_final - aep_floris_final) / aep_floris_final * 100))
            print("FLORIS Time:       {:.3f} s".format(time_floris_final))
            print("FLOWERS Time:      {:.3f} s".format(time_flowers_final))
            print("Factor of Improvement: {:.1f}x".format(time_floris_final/time_flowers_final))
            print("============================")
        else:
            return (aep_flowers_final, aep_floris_final), (time_flowers_final, time_floris_final)

    ###########################################################################
    # Optimization methods
    ###########################################################################

    def initialize_optimization(self, boundaries, num_terms=None, wd_resolution=1.0):
        """
        Initialize FLOWERS and FLORIS interfaces for layout optimization.
        FLORIS optimization is performed with average wind speeds for each
        wind direction.

        Args:
            boundaries (list(tuple)): boundary vertices in the form
                [(x0,y0), (x1,y1), ... , (xN,yN)]
            num_terms (int, optional): for FLOWERS, the number of Fourier modes
                to compute AEP in the range [1, ceiling(num_wind_directions/2)].
                Defaults to the maximum number.
            wd_resolution (float, optional): for FLORIS, the width of the discrete 
                wind direction bins to compute AEP. Defaults to the original
                resolution of the wind rose.
        
        Returns:
            self.flowers: FLOWERS interface
            self.floris: FLORIS interface
        """

        print("Initializing optimization problem with FLOWERS and FLORIS-" + self.model.capitalize() + ".")

        # Add boundary as class member and save boundary limits for normalization
        self.boundaries = boundaries
        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])

        # Compute initial AEP with FLORIS post-processor
        self.aep_initial = self.post.get_farm_AEP(freq=self.post_freq)

        # Reinitialize FLOWERS to user specification
        if num_terms == None:
            self.reinitialize_flowers()
        else:
            self.truncate_flowers(num_terms=num_terms)
        
        # Reinitialize FLORIS to user specification
        self.resample_floris(ws_avg=True, wd_resolution=wd_resolution)
        self.floris.calculate_wake()

        # Set initialization flag to True
        self.opt_init = True

        return self.flowers, self.floris

    def run_flowers_optimization(
        self, 
        history_file='',
        output_file='',
        timer=86400,
        o_tol=1e-4,
        f_tol=1e-4,
        ):
        """
        Run layout optimization with FLOWERS. 

        Args:
            solver (str): name of PyOptSparse optimization algorithm.
                Currently supported choices: SNOPT
            history_file (str): file name of PyOptSparse_History output file.
                Should end with '.hist'
            output_file (str): file name of PyOptSparse output file.
                Should end with '.out'
            timer (int, optional): wall-time limit for optimizer [s].
                Defaults to 1 day.

        Returns:
            flowers_solution (dict): stores the quantities of interest from FLOWERS 
                optimization results:
                - 'iter' (int): number of major iterations
                - 'aep' (list(float)): AEP (post-processed with 
                    specific FLORIS interface) of layout at each major iteration
                - 'layout' (tuple(numpy.array)): container of farm layout at each
                    iteration. Calling layout[x/y][iteration] will give a numpy array
                    of the x- or y-positions of each turbine at that iteration.
                - 'opt' (list(float)): SNOPT optimality at each major iteration
                - 'feas' (list(float)): SNOPT feasibility at each major iteration
                - 'con_bound' (numpy.array): 2D array of the boundary constraint for
                    each turbine. Calling con_bound[i] outputs the array of the 
                    distance to the farm boundary (with negative being inside the 
                    boundary) for turbine 'i'.
                - 'con_spacing' (list(float)): spacing constraint value at each
                    major iteration.
                - 'time' (float): wall-time of optimization study
                - 'solver_time' (float): time of solver (?)
                - 'obj_calls' (int): number of AEP function calls
            layout_flowers (tuple(numpy.array)): array of the x- or
                y-position of each turbine (x: 0; y: 1)
            aep_flowers (float): farm AEP [Wh]
        
        """

        print("Solving FLOWERS optimization.")

        # Verify required file names
        if not history_file:
            raise ValueError('History file name must be specified.')

        if not output_file:
            raise ValueError('Output file name must be specified.')

        # Adjust verbose output file name
        verbose_file = output_file[:-3] + 'verb'

        # Verify interface has been initialized
        if not self.opt_init:
            raise RuntimeError("Optimization has not been initialized.")

        # Initialize layout optimization class
        model = lyt.LayoutOptimization(
            self.flowers, 
            self.boundaries,
            storeHistory=history_file,
            optOptions={
                'Print file': verbose_file, 
                'Summary file': output_file,
                "Major optimality tolerance": o_tol,
                "Major feasibility tolerance": f_tol,
                "Scale option": 2,
                },
            timeLimit=timer,
            )

        # Run solver and post-process results
        sol = model.optimize()
        self._save_flowers_solution(sol, history_file)
        self.opt_flowers = True
    
    def run_floris_optimization(
        self, 
        solver='SNOPT',
        history_file='',
        output_file='',
        timer=86400,
        o_tol=1e-4,
        f_tol=1e-4,
        ):
        """
        Run layout optimization with FLORIS. 

        Args:
            solver (str): name of PyOptSparse optimization algorithm.
                Currently supported choices: SNOPT
            history_file (str): file name of PyOptSparse_History output file.
                Should end with '.hist'
            output_file (str): file name of PyOptSparse output file.
                Should end with '.out'
            timer (int, optional): wall-time limit for optimizer [s].
                Defaults to 1 day.

        Returns:
            floris_solution (dict): stores the quantities of interest from FLORIS 
                optimization results:
                - 'iter' (int): number of major iterations
                - 'aep' (list(float)): AEP (post-processed with 
                    specific FLORIS interface) of layout at each major iteration
                - 'layout' (tuple(numpy.array)): container of farm layout at each
                    iteration. Calling layout[x/y][iteration] will give a numpy array
                    of the x- or y-positions of each turbine at that iteration.
                - 'opt' (list(float)): SNOPT optimality at each major iteration
                - 'feas' (list(float)): SNOPT feasibility at each major iteration
                - 'con_bound' (numpy.array): 2D array of the boundary constraint for
                    each turbine. Calling con_bound[i] outputs the array of the 
                    distance to the farm boundary (with negative being inside the 
                    boundary) for turbine 'i'.
                - 'con_spacing' (list(float)): spacing constraint value at each
                    major iteration.
                - 'time' (float): wall-time of optimization study
                - 'solver_time' (float): time of solver (?)
                - 'obj_calls' (int): number of AEP function calls
            layout_floris (tuple(numpy.array)): array of the x- or
                y-position of each turbine (x: 0; y: 1)
            aep_floris (float): farm AEP [Wh]
        
        """

        print("Solving FLORIS optimization.")

        # Verify choice of solver
        solver_choices = ["SNOPT"]

        if solver not in solver_choices:
            raise ValueError(
                "Solver must be one supported by pyOptSparse: "
                + str(solver_choices)
            )

        # Verify required file names
        if not history_file:
            raise ValueError('History file name must be specified.')

        if not output_file:
            raise ValueError('Output file name must be specified.')

        # Adjust verbose output file name
        verbose_file = output_file[:-3] + 'verb'

        # Verify interface has been initialized
        if not self.opt_init:
            raise RuntimeError("Optimization has not been initialized.")

        # Initialize layout optimization class
        tmp = LayoutOptimizationPyOptSparse(
            self.floris,
            self.boundaries,
            freq=self.freq_floris,
            solver=solver,
            storeHistory=history_file,
            optOptions={
                'Print file': verbose_file, 
                'Summary file': output_file,
                "Major optimality tolerance": o_tol,
                "Major feasibility tolerance": f_tol,
                "Scale option": 2,
                },
            timeLimit=timer,
        )

        # Run solver and post-process results
        sol = tmp.optimize()
        self._save_floris_solution(sol, history_file)
        self.opt_floris = True
    
    def _save_flowers_solution(self, sol, history_file):
        """
        Private method to store the quantities of interest from the FLOWERS 
        optimization results in a dictionary flowers_solution:
            - 'iter' (int): number of major iterations
            - 'aep' (list(float)): AEP (post-processed with 
                specific FLORIS interface) of layout at each major iteration
            - 'layout' (tuple(numpy.array)): container of farm layout at each
                iteration. Calling layout[x/y][iteration] will give a numpy array
                of the x- or y-positions of each turbine at that iteration.
            - 'opt' (list(float)): SNOPT optimality at each major iteration
            - 'feas' (list(float)): SNOPT feasibility at each major iteration
            - 'con_bound' (numpy.array): 2D array of the boundary constraint for
                each turbine. Calling con_bound[i] outputs the array of the 
                distance to the farm boundary (with negative being inside the 
                boundary) for turbine 'i'.
            - 'con_spacing' (list(float)): spacing constraint value at each
                major iteration.
            - 'time' (float): wall-time of optimization study
            - 'solver_time' (float): time of solver (?)
            - 'obj_calls' (int): number of AEP function calls

        Also store the final solution:
            - layout_flowers (tuple(numpy.array)): array of the x- or
                y-position of each turbine (x: 0; y: 1)
            - aep_flowers (float): farm AEP [Wh]

        Args:
            sol: Optimization solution after running optimize()
            history (str): Name of history output file
        """

        print("Storing FLOWERS solution.")

        # Read history
        hist = History(history_file)
        # val = hist.getValues(
        #     names=[
        #         'feasibility',
        #         'optimality',
        #         'boundary_con',
        #         'spacing_con',
        #         'x',
        #         'y'
        #         ], 
        #         major=True
        #         )
        val = hist.getValues(
            names=[
                'x',
                'y',
            ],
            allowSens=True,
            callCounters=hist.callCounters,
            major=False
        )
        xx = val['x']
        yy = val['y']

        # Store history
        # TODO: store SNOPT exit code
        self.flowers_solution = dict()
        self.flowers_solution['iter'] = len(xx) - 1
        # self.flowers_solution['opt'] = val['optimality'].flatten()
        # self.flowers_solution['feas'] = val['feasibility'].flatten()
        # self.flowers_solution['con_bound'] = np.swapaxes(val['boundary_con'],0,1)
        # self.flowers_solution['con_space'] = val['spacing_con'].flatten()

        # # Store layouts
        # self.flowers_solution['layout'] = (self._unnorm(xx, self.xmin, self.xmax), self._unnorm(yy, self.ymin, self.ymax))
        # self.layout_flowers = (self.flowers_solution['layout'][0][-1],self.flowers_solution['layout'][1][-1])

        # # Compute and store AEP
        # self.flowers_solution['aep'] = []
        # for i in range(len(xx)):
        #     self.post.reinitialize(layout=(self.flowers_solution['layout'][0][i],self.flowers_solution['layout'][1][i]))
        #     self.flowers_solution['aep'].append(self.post.get_farm_AEP(freq=self.post_freq))
        # self.aep_flowers = self.flowers_solution['aep'][-1]
        xxx = self._unnorm(sol.getDVs()["x"], self.xmin, self.xmax)
        yyy = self._unnorm(sol.getDVs()["y"], self.ymin, self.ymax)
        self.layout_flowers = (xxx, yyy)
        self.post.reinitialize(layout=(xxx,yyy))
        self.aep_flowers = self.post.get_farm_AEP(freq=self.post_freq)

        # Store optimization performance
        self.flowers_solution['time'] = float(sol.optTime)
        self.flowers_solution['solver_time'] = float(sol.optCodeTime)
        self.flowers_solution['obj_calls'] = float(sol.userObjCalls)

    def _save_floris_solution(self, sol, history="hist.hist"):
        """
        Private method to store the quantities of interest from the FLORIS 
        optimization results in a dictionary floris_solution:
            - 'iter' (int): number of major iterations
            - 'aep' (list(float)): AEP (post-processed with 
                specific FLORIS interface) of layout at each major iteration
            - 'layout' (tuple(numpy.array)): container of farm layout at each
                iteration. Calling layout[x/y][iteration] will give a numpy array
                of the x- or y-positions of each turbine at that iteration.
            - 'opt' (list(float)): SNOPT optimality at each major iteration
            - 'feas' (list(float)): SNOPT feasibility at each major iteration
            - 'con_bound' (numpy.array): 2D array of the boundary constraint for
                each turbine. Calling con_bound[i] outputs the array of the 
                distance to the farm boundary (with negative being inside the 
                boundary) for turbine 'i'.
            - 'con_spacing' (list(float)): spacing constraint value at each
                major iteration.
            - 'time' (float): wall-time of optimization study
            - 'solver_time' (float): time of solver (?)
            - 'obj_calls' (int): number of AEP function calls

        Also store the final solution:
            - layout_floris (tuple(numpy.array)): array of the x- or
                y-position of each turbine (x: 0; y: 1)
            - aep_floris (float): farm AEP [Wh]

        Args:
            sol: Optimization solution after running optimize()
            history (str): Name of history output file
        """

        print("Storing FLORIS solution.")

        # Read history
        hist = History(history)
        # val = hist.getValues(
        #     names=[
        #         'feasibility',
        #         'optimality',
        #         'boundary_con',
        #         'spacing_con',
        #         'x',
        #         'y'
        #         ], 
        #         major=True
        #         )
        val = hist.getValues(
            names=[
                'x',
                'y',
            ],
            allowSens=True,
            callCounters=hist.callCounters,
            major=False
        )
        xx = val['x']
        yy = val['y']

        # Store history
        self.floris_solution = dict()
        self.floris_solution['iter'] = len(xx) - 1
        # self.floris_solution['opt'] = val['optimality'].flatten()
        # self.floris_solution['feas'] = val['feasibility'].flatten()
        # self.floris_solution['con_bound'] = np.swapaxes(val['boundary_con'],0,1)
        # self.floris_solution['con_space'] = val['spacing_con'].flatten()

        # # Store layouts
        # self.floris_solution['layout'] = (self._unnorm(xx, self.xmin, self.xmax), self._unnorm(yy, self.ymin, self.ymax))
        # self.layout_floris = (self.floris_solution['layout'][0][-1],self.floris_solution['layout'][1][-1])

        # # Compute and store AEP
        # self.floris_solution['aep'] = []
        # for i in range(len(xx)):
        #     self.post.reinitialize(layout=(self.floris_solution['layout'][0][i],self.floris_solution['layout'][1][i]))
        #     self.floris_solution['aep'].append(self.post.get_farm_AEP(freq=self.post_freq))
        # self.aep_floris = self.floris_solution['aep'][-1]
        xxx = self._unnorm(sol.getDVs()["x"], self.xmin, self.xmax)
        yyy = self._unnorm(sol.getDVs()["y"], self.ymin, self.ymax)
        self.layout_floris = (xxx, yyy)
        self.post.reinitialize(layout=(xxx,yyy))
        self.aep_floris = self.post.get_farm_AEP(freq=self.post_freq)

        # Store optimization performance
        self.floris_solution['time'] = float(sol.optTime)
        self.floris_solution['solver_time'] = float(sol.optCodeTime)
        self.floris_solution['obj_calls'] = float(sol.userObjCalls)

    def show_flowers_feasibility(self, inf=False):

        feasible=True
        viol = {}
        # Spacing constraint
        x = self.layout_flowers[0]
        y = self.layout_flowers[1]

        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)
        if np.min(dist) <= 1.99 * self.diameter:
            feasible = False
            viol['Spacing [D]'] = np.min(dist)/self.diameter
        
        # Boundary constraint
        boundary_polygon = Polygon(self.boundaries)
        boundary_line = LineString(self.boundaries)
        boundary_con = np.zeros(len(x))
        for i in range(len(x)):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(boundary_line) #NaNsafe, or 1 to 5 m inside boundary
            if boundary_polygon.contains(loc)==True:
                boundary_con[i] *= -1.0
        if np.max(boundary_con) >  1:
            feasible = False
            viol['Boundary [m]'] = np.max(boundary_con)
        
        # Output
        if inf:
            return viol
        else:
            return feasible
    
    def show_floris_feasibility(self, inf=False):

        feasible=True
        viol = {}
        # Spacing constraint
        x = self.layout_floris[0]
        y = self.layout_floris[1]

        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)
        if np.min(dist) <= 1.99 * self.diameter:
            feasible = False
            viol['Spacing [D]'] = np.min(dist)/self.diameter
        
        # Boundary constraint
        boundary_polygon = Polygon(self.boundaries)
        boundary_line = LineString(self.boundaries)
        boundary_con = np.zeros(len(x))
        for i in range(len(x)):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(boundary_line) #NaNsafe, or 1 to 5 m inside boundary
            if boundary_polygon.contains(loc)==True:
                boundary_con[i] *= -1.0
        if np.max(boundary_con) >  1:
            feasible = False
            viol['Boundary [m]'] = np.max(boundary_con)
        
        # Output
        if inf:
            return viol
        else:
            return feasible

    def show_optimization_comparison(self, stats=False):
        """
        Compare optimization solutions of both FLOWERS and FLORIS.
        Outputs a table of optimal AEP and optimizer performance to
        the terminal.

        Args:
            stats (bool, optional): dictates whether optimizer
                performance (iterations, time, function evaluations)
                are output. Defaults to False.

        """

        if not self.opt_floris or not self.opt_flowers:
            raise RuntimeError("Both optimizations have not been run.")
        
        print("============================")
        print('    Optimization Results    ')
        print('    FLOWERS Terms: {:.0f}'.format(self.terms_flowers))
        print('    FLORIS Bins:   {:.0f}'.format(self.bins_floris))
        print('    FLORIS Model: ' + self.model.capitalize())
        print("----------------------------")
        print("Initial AEP:      {:.3f} GWh".format(self.aep_initial / 1.0e9))
        print("FLORIS  AEP:      {:.3f} GWh".format(self.aep_floris / 1.0e9))
        print("FLOWERS AEP:      {:.3f} GWh".format(self.aep_flowers / 1.0e9))
        print("FLORIS AEP Gain:       {:.2f}%".format((self.aep_floris - self.aep_initial) / self.aep_initial * 100))
        print("FLOWERS AEP Gain:      {:.2f}%".format((self.aep_flowers - self.aep_initial) / self.aep_initial * 100))

        if stats:
            print("----------------------------")
            print("FLOWERS Time:         {:.1f} s".format(self.flowers_solution['time']))
            print("FLORIS Time:          {:.1f} s".format(self.floris_solution['time']))
            print("Speed-Up Factor:      {:.2f}x".format(self.floris_solution['time']/self.flowers_solution['time']))
            print()
            print("FLOWERS AEP Evaluations:  {:.0f}".format(self.flowers_solution['obj_calls']))
            print("FLORIS AEP Evaluations:   {:.0f}".format(self.floris_solution['obj_calls']))
            print("FLOWERS Iterations:       {:.0f}".format(self.flowers_solution['iter']))
            print("FLORIS Iterations:        {:.0f}".format(self.floris_solution['iter']))
        
        print("============================")
    
    def show_flowers_optimization(self, stats=False):
        """
        Show the optimization results of FLOWERS. Outputs a table 
        of optimal AEP and optimizer performance to the terminal.

        Args:
            stats (bool, optional): dictates whether optimizer
                performance (iterations, time, function evaluations)
                are output. Defaults to False.

        """

        if not self.opt_flowers:
            raise RuntimeError("The FLOWERS optimization has not been run.")

        print("============================")
        print('    Optimization Results    ')
        print('    FLOWERS Terms: {:.0f}'.format(self.terms_flowers))
        print("----------------------------")
        print("Initial AEP:      {:.3f} GWh".format(self.aep_initial / 1.0e9))
        print("Optimal AEP:      {:.3f} GWh".format(self.aep_flowers / 1.0e9))
        print("FLOWERS AEP Gain:      {:.2f}%".format((self.aep_flowers - self.aep_initial) / self.aep_initial * 100))

        if stats:
            print("----------------------------")
            print("FLOWERS Time:         {:.1f} s".format(self.flowers_solution['time']))
            print("FLOWERS AEP Evaluations:  {:.0f}".format(self.flowers_solution['obj_calls']))
            print("FLOWERS Iterations:       {:.0f}".format(self.flowers_solution['iter']))
        
        print("============================")

    def show_floris_optimization(self, stats=False):
        """
        Show the optimization results of FLORIS. Outputs a table 
        of optimal AEP and optimizer performance to the terminal.

        Args:
            stats (bool, optional): dictates whether optimizer
                performance (iterations, time, function evaluations)
                are output. Defaults to False.

        """

        if not self.opt_floris:
            raise RuntimeError("The FLORIS optimization has not been run.")

        print("============================")
        print('    Optimization Results    ')
        print('    FLORIS Bins: {:.0f}'.format(self.bins_floris))
        print('    FLORIS Model: ' + self.model.capitalize())
        print("----------------------------")
        print("Initial AEP:      {:.3f} GWh".format(self.aep_initial / 1.0e9))
        print("Optimal AEP:      {:.3f} GWh".format(self.aep_floris / 1.0e9))
        print("FLORIS AEP Gain:      {:.2f}%".format((self.aep_floris - self.aep_initial) / self.aep_initial * 100))

        if stats:
            print("----------------------------")
            print("FLORIS Time:         {:.1f} s".format(self.floris_solution['time']))
            print("FLORIS AEP Evaluations:  {:.0f}".format(self.floris_solution['obj_calls']))
            print("FLORIS Iterations:       {:.0f}".format(self.floris_solution['iter']))
        
        print("============================")

    def plot_flowers_layout(self, ax=None, color_initial="tab:blue", color_final="tab:orange"):
        """
        Plot the initial and optimal layouts of the FLOWERS
        optimization.

        Args:
            ax_aep (:py:class:`matplotlib.pyplot.axes`, optional): axis to
                plot layouts

        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        vis.plot_optimal_layout(
            ax=ax, 
            boundaries=self.boundaries, 
            x_final=self.layout_flowers[0], 
            y_final=self.layout_flowers[1], 
            x_init=self.layout_x, 
            y_init=self.layout_y, 
            D=self.diameter,
            color_initial=color_initial,
            color_final=color_final
        )
        # plt.legend(
        #     ["Initial", "Final"],
            #bbox_to_anchor=(0.05, 1.12),
        # )
        ax.set(title="FLOWERS")
        return ax
    
    def plot_floris_layout(self, ax=None, color_initial="tab:blue", color_final="tab:orange"):
        """
        Plot the initial and optimal layouts of the FLORIS
        optimization.

        Args:
            ax_aep (:py:class:`matplotlib.pyplot.axes`, optional): axis to
                plot layouts

        """
        if ax is None:
            _, ax = plt.subplots(1,1)

        vis.plot_optimal_layout(
            ax=ax, 
            boundaries=self.boundaries, 
            x_final=self.layout_floris[0], 
            y_final=self.layout_floris[1], 
            x_init=self.layout_x, 
            y_init=self.layout_y, 
            D=self.diameter,
            color_initial=color_initial,
            color_final=color_final
        )
        # plt.legend(
        #     ["Initial", "Final"],
            #bbox_to_anchor=(0.05, 1.12),
        # )
        ax.set(title="FLORIS")
        return ax
    
    def plot_optimal_layouts(self):
        """
        Plot the initial and optimal layouts of both the
        FLOWERS and FLORIS optimization.

        """

        fig, (ax0, ax1) = plt.subplots(1,2)
        vis.plot_optimal_layout(
            ax=ax0, 
            boundaries=self.boundaries, 
            x_final=self.layout_flowers[0], 
            y_final=self.layout_flowers[1], 
            x_init=self.layout_x, 
            y_init=self.layout_y, 
            D=self.diameter,
        )
        vis.plot_optimal_layout(
            ax=ax1, 
            boundaries=self.boundaries, 
            x_final=self.layout_floris[0], 
            y_final=self.layout_floris[1], 
            x_init=self.layout_x, 
            y_init=self.layout_y, 
            D=self.diameter,
        )
        ax0.set_title('FLOWERS')
        ax1.set_title('FLORIS')
        fig.tight_layout()

        # plt.legend(
        #     ["Initial", "Final"],
            #bbox_to_anchor=(0.38, 1.1),
            #ncol=2,
        # )
    
    def plot_flowers_history(self, animate_file=None, show=False):
        """
        Plot the AEP, optimality, and feasibility history of the
        FLOWERS optimization.
    
        Args:
            animate_file (str, optional): file name for layout animation,
                if desired.
            show (bool, optional): dictates whether the layout animation is
                displayed before closing. Defaults to False.

        """

        fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(12,4.5))
        vis.plot_convergence_history(
            aep=self.flowers_solution['aep'],
            optimality=self.flowers_solution['opt'],
            feasibility=self.flowers_solution['feas'],
            ax_aep=ax0,
            ax_opt=ax1,
            ax_feas=ax2
        )
        fig.suptitle("FLOWERS")
        fig.tight_layout()

        if animate_file is not None:
            vis.animate_layout_history(
                filename=animate_file,
                layout_x=self.flowers_solution['layout'][0],
                layout_y=self.flowers_solution['layout'][1],
                boundaries=self.boundaries,
                D=self.diameter,
                show=show,
            )
    
    def plot_floris_history(self, animate_file=None, show=False):
        """
        Plot the AEP, optimality, and feasibility history of the
        FLORIS optimization.
    
        Args:
            animate_file (str, optional): file name for layout animation,
                if desired.
            show (bool, optional): dictates whether the layout animation is
                displayed before closing. Defaults to False.

        """

        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        vis.plot_convergence_history(
            aep=self.floris_solution['aep'],
            optimality=self.floris_solution['opt'],
            feasibility=self.floris_solution['feas'],
            ax_aep=ax0,
            ax_opt=ax1,
            ax_feas=ax2
        )
        fig.suptitle("FLORIS")
        fig.tight_layout()

        if animate_file is not None:
            vis.animate_layout_history(
                filename=animate_file,
                layout_x=self.floris_solution['layout'][0],
                layout_y=self.floris_solution['layout'][1],
                boundaries=self.boundaries,
                D=self.diameter,
                show=show,
            )
    
    def plot_optimization_histories(self, flowers_mov=None, floris_mov=None):
        """
        Plot the AEP, optimality, and feasibility history of the
        FLOWERS and FLORIS optimization.
    
        Args:
            flowers_mov (str, optional): file name for FLOWERS layout animation,
                if desired.
            floris_mov (str, optional): file name for FLORIS layout animation,
                if desired.

        """

        fig, ((ax0, ax1, ax2),(ax3, ax4, ax5)) = plt.subplots(2,3)
        
        vis.plot_convergence_history(
            aep=self.floris_solution['aep'],
            optimality=self.floris_solution['opt'],
            feasibility=self.floris_solution['feas'],
            ax_aep=ax0,
            ax_opt=ax1,
            ax_feas=ax2
        )
        vis.plot_convergence_history(
            aep=self.floris_solution['aep'],
            optimality=self.floris_solution['opt'],
            feasibility=self.floris_solution['feas'],
            ax_aep=ax3,
            ax_opt=ax4,
            ax_feas=ax5
        )
        ax1.set_title("FLOWERS")
        ax4.set_title("FLORIS")
        fig.tight_layout()

        if flowers_mov is not None:
            vis.animate_layout_history(
                filename=flowers_mov,
                layout_x=self.flowers_solution['layout'][0],
                layout_y=self.flowers_solution['layout'][1],
                boundaries=self.boundaries,
                D=self.diameter,
                show=False,
            )
            vis.animate_layout_history(
                filename=floris_mov,
                layout_x=self.floris_solution['layout'][0],
                layout_y=self.floris_solution['layout'][1],
                boundaries=self.boundaries,
                D=self.diameter,
                show=False,
            )

    ###########################################################################
    # Internal tools
    ###########################################################################

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1