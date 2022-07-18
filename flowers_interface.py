# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd
from pyoptsparse.pyOpt_history import History
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt

import floris.tools as wfct

import tools as tl
import visualization as vis


class Flowers():
    """
    Flowers is a high-level user interface to the underlying
    methods within the FLOWERS framework. It is an entry-point for users
    to simplify calls to methods on objects within FLOWERS.

    Args:
        wind_rose (pandas:dataframe): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws': np.array of wind speeds
                - 'wd': np.array of wind directions
                - 'freq_val': np.array of frequency for each wind speed and direction
        layout_x (numpy:array): Array of x-positions of each turbine
        layout_y (numpy:array): Array of y-positions of each turbine
        k (float): wake expansion rate
        D (float): rotor diameter
        fs (pandas:dataframe): A container for the Fourier coefficients used to
            expand the wind rose:
                - 'a_free':
                - 'b_free':
                - 'a_wake':
                - 'b_wake':
    """

    def __init__(self, wind_rose, layout_x, layout_y, k=0.05, D=126):
        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.k = k
        self.D = D

    def fourier_coefficients(self, num_terms=0, wd_resolution=1.0):
        """Compute Fourier series expansion coefficients from wind rose"""

        # Resample wind rose for average wind speed per wind direction
        wr = self.wind_rose.copy(deep=True)
        
        # TODO: reintroduce coarser wind rose to calculate FS coefficients
        # wr = tl.resample_wind_direction(wr, wd=wd)
        wr = tl.resample_average_ws_by_wd(wr)

        # Transform wind direction to polar angle
        wr["wd"] = np.remainder(450 - wr.wd, 360)
        wr.sort_values("wd", inplace=True)

        # Look up thrust coefficient for each wind direction bin
        ct = np.zeros(len(wr.ws))
        for wd in range(len(wr.ws)):
            ct[wd] = tl.ct_lookup(wr.ws[wd])
            if ct[wd] >= 1.0:
                ct[wd] = 0.99999
        
        # Fourier expansion of freestream term
        g = 1 / (2 * np.pi) * wr.ws * wr.freq_val
        gft = 2 * np.fft.rfft(g)
        a_free =  gft.real
        b_free = -gft.imag

        # Fourier expansion of wake deficit term
        h = 1 / (2 * np.pi) * (1 - np.sqrt(1 - ct)) * wr.ws * wr.freq_val
        hft = 2 * np.fft.rfft(h)
        a_wake =  hft.real
        b_wake = -hft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a_free):
            a_free = a_free[0:num_terms]
            b_free = b_free[0:num_terms]
            a_wake = a_wake[0:num_terms]
            b_wake = b_wake[0:num_terms]

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a_free': a_free, 'b_free': b_free, 'a_wake': a_wake, 'b_wake': b_wake})

    def calculate_wake(self, x, y):
        """
        Compute the annually-averaged velocity deficit at a given set of points.

        Args:
            x: relative x-position of points where wake should be calculated
                (with turbine location at the origin); can be a single float
                when computing power or an array when computing a flow field
            y: relative y-position of points where wake should be calculated
                (with turbine location at the origin); can be a single float
                when computing power or an array when computing a flow field
            num_terms: number of Fourier series terms to use in computation.
                Defaults to maximum number permitted based on wind rose resolution.
        
        Returns:
            du: average wake velocity deficit at the given position(s) (x,y)
        """

        # Enforce that fourier_coefficients() have been computed
        if not hasattr(self, 'fs'):
            print(
                "Error, must compute Fourier coefficients before calculating wake"
            )
            return None

        # Normalize positions by rotor radius
        x /= self.D/2
        y /= self.D/2

        # Convert to polar coordinates
        R = np.sqrt(x**2 + y**2)
        THETA = np.arctan2(y,x) + np.pi

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.abs(np.arctan(
            (1 / R + self.k * np.sqrt(1 + self.k**2 - R**(-2)))
            / (-self.k / R + np.sqrt(1 + self.k**2 - R**(-2)))
            ))

        # Wake velocity contribution from each Fourier mode
        # TODO: enforce maximum wake velocity deficit if R < 1
        if R < 1:
            du = 0.9999
        else:
            du = self.fs.a_wake[0] * (
                theta_c * (self.k * R * (theta_c**2 + 3) + 3) / (3 * (self.k * R + 1)**3)
                )
            # TODO: eliminate for loop with matrix multiplication
            for n in range(1, len(self.fs.b_wake)):
                du += (2 * (
                    self.fs.a_wake[n] * np.cos(n * THETA) + self.fs.b_wake[n] * np.sin(n * THETA))
                    ) / (n * (self.k * R + 1))**3 * (
                        np.sin(n * theta_c) 
                        * (n**2 * (self.k * R * (theta_c**2 + 1) + 1) - 2 * self.k * R) 
                        + 2 * n * self.k * R * theta_c * np.cos(n * theta_c)
                        )

        return du

    def calculate_aep(self):
        """
        Compute AEP from FLOWERS interface based on given wind rose and
        wind farm layout. Returns AEP [Wh].
        """

        # Power component from freestream
        p0 = self.fs.a_free[0] * np.pi
        
        # Sum AEP contribution of each turbine
        aep = 0.
        # TODO: avoid 2 for loops
        for i in range(len(self.layout_x)):

            # Define array of turbine position relative to turbine 'i'
            X = self.layout_x[i] - self.layout_x
            Y = self.layout_y[i] - self.layout_y

            # Remove turbine to prevent computing wake interaction with itself
            X = np.delete(X,i)
            Y = np.delete(Y,i)

            # Compute power from each pairwise turbine interaction
            p = 0.
            for j in range(len(X)):
                p += self.calculate_wake(X[j],Y[j])
            
            # Look up power coefficient from average wake velocity
            aep += tl.cp_lookup(p0 - p)  * (p0 - p)**3
        
        return 0.5 * 1.225 * np.pi * self.D**2 / 4 * 8760 * aep

class ModelComparison:
    """
    ModelComparison is an interface for comparing AEP and optimization
    performance between FLOWERS and FLORIS. It streamlines the initialization
    of FLOWERS and FLORIS interfaces.

    Args:
        wind_rose (pandas:dataframe): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws': np.array of wind speeds
                - 'wd': np.array of wind directions
                - 'freq_val': np.array of frequency for each wind speed and direction
        layout_x (numpy:array): Array of x-positions of each turbine
        layout_y (numpy:array): Array of y-positions of each turbine
    """

    def __init__(self, wind_rose, layout_x, layout_y, model="jensen"):
        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.model = model

        # Initialize FLORIS from Jensen input file
        if model == "jensen":
            input_file = "./input/jensen.yaml"
        elif model == "gauss":
            input_file = "./input/gauss.yaml"
        self.floris = wfct.floris_interface.FlorisInterface(input_file)
        self.floris.reinitialize(
            layout=(layout_x.flatten(),layout_y.flatten()), 
            wind_shear=0)
        
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

        # TODO: import from FLORIS
        k = 0.05 
        self.diameter = self.floris.floris.farm.rotor_diameters[0][0][0]

        # Initialize FLOWERS interface
        self.flowers = Flowers(
            self.wind_rose,
            self.layout_x,
            self.layout_y,
            k=k,
            D=self.diameter
            )

    ###########################################################################
    # Internal tools
    ###########################################################################

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    ###########################################################################
    # Initialization methods
    ###########################################################################

    def get_flowers_interface(self):
        """Get current instantiation of FLOWERS interface"""
        return self.flowers

    def get_floris_interface(self):
        """Get current instantiation of FLORIS interface"""
        return self.floris

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
        """

        # Resample wind rose with specified wind direction bins
        if wd_resolution > 1.0:
            wr = tl.resample_wind_direction(self.wind_rose, wd=np.arange(0, 360, wd_resolution))
        else:
            wr = self.wind_rose

        # Resample wind rose by average wind speed per wind direction
        if ws_avg:
            wr = tl.resample_average_ws_by_wd(wr)
            freq = wr.freq_val.to_numpy()
            self.freq_floris = freq / np.sum(freq)
            self.bins_floris = len(wr.wd)

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

    def initialize_optimization(self, boundaries=None, num_terms=None, wd_resolution=None):
        """
        Initialize FLOWERS and FLORIS interfaces for optimization.

        Args:
            boundaries (list(float)): A list of the boundary vertices in the form
                [(x0,y0), (x1,y1), ... , (xN,yN)]
            num_terms (int): Number of Fourier modes for the FLOWERS solution
            wd_resolution (float): Width of the wind rose bins for the FLORIS solution
        
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

        # Reinitialize FLORIS interface and compute initial AEP
        self.reinitialize_floris()
        self.aep_initial = self.floris.get_farm_AEP(freq=self.freq_floris)

        # Reinitialise FLOWERS to user specification
        if num_terms == None:
            self.reinitialize_flowers()
        else:
            self.truncate_flowers(num_terms=num_terms)
        
        # Reinitialize FLORIS to user specification
        if wd_resolution != None:
            self.resample_floris(ws_avg=True, wd_resolution=wd_resolution)

        return self.flowers, self.floris
    
    ###########################################################################
    # Store optimization solutions
    ###########################################################################
    
    def save_flowers_solution(self, sol, history="hist.hist"):
        """
        Store the quantities of interest from the FLOWERS optimization results
        in a dictionary flowers_solution:
            ** Number of iterations ['iter']
            ** History of major objective evaluations ['aep']
            ** History of wind farm (x,y) layout ['layout']
            ** Optimization time ['time']
            ** Solver time ['solver_time']
            ** Number of objective calls ['obj_calls']

        Also store optimal layout (flowers_layout) and optimal AEP (aep_flowers)

        Args:
            sol: Optimization solution after running optimize()
            history (str): Name of history output file
                default: "hist.hist"
        """

        print("Storing FLOWERS solution.")

        # Read layout history
        hist = History(history)
        val = hist.getValues(names=['feasibility','optimality','x','y'], major=True)
        xx = val['x']
        yy = val['y']

        # TODO: Save major iterations
        # obj = val['obj']
        # obj1 = np.minimum.accumulate(obj).flatten()
        # _, ind = np.unique(obj1, return_index=True)
        # _, ind = np.unique(np.minimum.accumulate(obj).flatten(), return_index=True)
        # ind = np.flip(ind[1:])
        # print(ind)
        # ind = range(len(obj))

        # Store solution data
        self.flowers_solution = dict()
        self.flowers_solution['iter'] = len(xx) - 1
        self.flowers_solution['opt'] = val['optimality'].flatten()
        self.flowers_solution['feas'] = val['feasibility'].flatten()

        # Store layouts
        self.flowers_solution['layout'] = (self._unnorm(xx, self.xmin, self.xmax), self._unnorm(yy, self.ymin, self.ymax))
        self.flowers_layout = (self.flowers_solution['layout'][0][-1],self.flowers_solution['layout'][1][-1])
        # Store AEP
        self.flowers_solution['aep'] = []
        self.reinitialize_floris()
        for i in range(len(xx)):
            self.floris.reinitialize(layout=(self.flowers_solution['layout'][0][i],self.flowers_solution['layout'][1][i]))
            self.flowers_solution['aep'].append(self.floris.get_farm_AEP(freq=self.freq_floris))
        self.aep_flowers = self.flowers_solution['aep'][-1]

        # Store optimization performance
        self.flowers_solution['time'] = float(sol.optTime)
        self.flowers_solution['solver_time'] = float(sol.optCodeTime)
        self.flowers_solution['obj_calls'] = float(sol.userObjCalls)


    def save_floris_solution(self, sol, history="hist.hist"):
        """
        Store the quantities of interest from the FLORIS optimization results
        in a dictionary floris_solution:
            ** Number of iterations ['iter']
            ** History of major objective evaluations ['aep']
            ** History of wind farm (x,y) layout ['layout']
            ** Optimization time ['time']
            ** Solver time ['solver_time']
            ** Number of objective calls ['obj_calls']

        Also store optimal layout (floris_layout) and optimal AEP (aep_floris)

        Args:
            sol: Optimization solution after running optimize()
            history (str): Name of history output file
                default: "hist.hist"
        """

        print("Storing FLORIS solution.")

        # Read layout history
        hist = History(history)
        val = hist.getValues(names=['feasibility','optimality','x','y'], major=True)
        xx = val['x']
        yy = val['y']

        # TODO: Save major iterations
        # obj = val['obj']
        # obj1 = np.minimum.accumulate(obj).flatten()
        # _, ind = np.unique(obj1, return_index=True)
        # _, ind = np.unique(np.minimum.accumulate(obj).flatten(), return_index=True)
        # ind = np.flip(ind[1:])
        # ind = range(len(obj))

        # Store solution data
        self.floris_solution = dict()
        self.floris_solution['iter'] = len(xx) - 1
        self.floris_solution['opt'] = val['optimality'].flatten()
        self.floris_solution['feas'] = val['feasibility'].flatten()

        # Store layouts
        self.floris_solution['layout'] = (self._unnorm(xx, self.xmin, self.xmax), self._unnorm(yy, self.ymin, self.ymax))
        self.floris_layout = (self.floris_solution['layout'][0][-1],self.floris_solution['layout'][1][-1])
        # Store AEP
        self.floris_solution['aep'] = []
        self.reinitialize_floris()
        for i in range(len(xx)):
            self.floris.reinitialize(layout=(self.floris_solution['layout'][0][i],self.floris_solution['layout'][1][i]))
            self.floris_solution['aep'].append(self.floris.get_farm_AEP(freq=self.freq_floris))
        self.aep_floris = self.floris_solution['aep'][-1]

        # Store optimization performance
        self.floris_solution['time'] = float(sol.optTime)
        self.floris_solution['solver_time'] = float(sol.optCodeTime)
        self.floris_solution['obj_calls'] = float(sol.userObjCalls)

    ###########################################################################
    # Compare and visualize results
    ###########################################################################

    def compare_aep(self, num_terms=None, wd_resolution=None):
        """
        Compare AEP evaluated with FLOWERS and FLORIS for the stored
        wind rose and wind farm layout. Reinitializes the FLOWERS and
        FLORIS interfaces stored in the class.

        Args:
            num_terms (int): Number of Fourier modes for the FLOWERS solution
            wd_resolution (float): Width of the wind rose bins for the FLORIS solution

        Returns: AEP results printed to terminal.
        """
        # Reinitialize FLOWERS and FLORIS interfaces
        if num_terms == None:
            self.reinitialize_flowers()
        else:
            self.truncate_flowers(num_terms=num_terms)
        
        if wd_resolution == None:
            self.reinitialize_floris()
        else:
            self.resample_floris(wd_resolution=wd_resolution)

        # Calculate AEP for both models
        aep_flowers = self.flowers.calculate_aep()
        aep_floris = self.floris.get_farm_AEP(freq=self.freq_floris)

        print("---------------------------")
        print("FLORIS  AEP:  {:.3f} GWh".format(aep_floris / 1.0e9))
        print("FLOWERS AEP:  {:.3f} GWh".format(aep_flowers / 1.0e9))
        print("Percent Diff: {:.2f}%".format((aep_flowers - aep_floris) / aep_floris * 100))
        print("---------------------------")

    def show_optimization_comparison(self, stats=False):
        """
        Compare AEP (evaluated with FLORIS at highest wind rose resolution)
        of the stored FLOWERS and FLORIS solutions. Reinitializes
        the FLORIS interface to compute AEP.

        Args:
            stats (bool): Include optimization information
                (time, function evaluations, number of iterations)

        Returns: Optimization results printed to terminal
        """

        print("============================")
        print('    Optimization Results    ')
        print('    FLOWERS Terms: {:.0f}'.format(self.terms_flowers))
        print('    FLORIS Bins:   {:.0f}'.format(self.bins_floris))
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
    
    def show_flowers_solution(self, stats=False):
        """Print quantities of interest of FLOWERS optimization"""
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

    def show_floris_solution(self, stats=False):
        """Print quantities of interest of FLOWERS optimization"""
        print("============================")
        print('    Optimization Results    ')
        print('    FLORIS Bins: {:.0f}'.format(self.bins_floris))
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

    def plot_flowers_layout(self, ax=None):
        """Plot initial and FLOWERS optimal layouts on specified axes"""
        if ax is None:
            _, ax = plt.subplots(1,1)
        vis.plot_optimal_layout(ax, self.boundaries, self.flowers_layout[0], self.flowers_layout[1], self.layout_x, self.layout_y, self.diameter)
        plt.legend(
            ["Old locations", "New locations"],
            bbox_to_anchor=(0.05, 1.12),
        )
        ax.set(title="FLOWERS")
        return ax
    
    def plot_floris_layout(self, ax=None):
        """Plot initial and FLORIS optimal layouts on specified axes"""
        if ax is None:
            _, ax = plt.subplots(1,1)
        vis.plot_optimal_layout(ax, self.boundaries, self.floris_layout[0], self.floris_layout[1], self.layout_x, self.layout_y, self.diameter)
        plt.legend(
            ["Old locations", "New locations"],
            bbox_to_anchor=(0.05, 1.12),
        )
        ax.set(title="FLORIS")
        return ax
    
    def plot_optimal_layouts(self):
        """Plot initial, FLOWERS, and FLORIS optimal layouts on specified axes"""

        _, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
        vis.plot_optimal_layout(ax0, self.boundaries, self.flowers_layout[0], self.flowers_layout[1], self.layout_x, self.layout_y, self.diameter)
        vis.plot_optimal_layout(ax1, self.boundaries, self.floris_layout[0], self.floris_layout[1], self.layout_x, self.layout_y, self.diameter)
        plt.legend(
            ["Old locations", "New locations"],
            bbox_to_anchor=(0.38, 1.1),
            ncol=2,
        )
        ax0.set_title('FLOWERS')
        ax1.set_title('FLORIS')
    
    def plot_flowers_history(self, ax=None, animate=None):
        """Plot history of FLOWERS optimization AEP and layout (optional)"""
        if ax is None:
            _, ax = plt.subplots(1,1)
        vis.plot_history(ax, self.flowers_solution['aep'], self.flowers_solution['layout'], self.boundaries, self.diameter, "flowers.mp4", animate)
        ax.set(title="FLOWERS")
        return ax
    
    def plot_floris_history(self, ax=None, animate=None):
        """Plot history of FLORIS optimization AEP and layout (optional)"""
        if ax is None:
            _, ax = plt.subplots(1,1)
        vis.plot_history(ax, self.floris_solution['aep'], self.floris_solution['layout'], self.boundaries, self.diameter, "floris.mp4", animate)
        ax.set(title="FLORIS")
        return ax
    
    def plot_optimization_histories(self, flowers_mov=None, floris_mov=None):
        """
        Plot history of FLOWERS and FLORIS optimizations, including objective function
            convergence and animations of layout development.
        """
        _, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
        vis.plot_history(ax0, self.flowers_solution['aep'], self.flowers_solution['layout'], self.boundaries, self.diameter, flowers_mov, show=False)
        vis.plot_history(ax1, self.floris_solution['aep'], self.floris_solution['layout'], self.boundaries, self.diameter, floris_mov, show=False)
        ax0.set_title('FLOWERS')
        ax1.set_title('FLORIS')