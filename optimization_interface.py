import numpy as np
import pyoptsparse
from shapely.geometry import Polygon, Point, LineString

class LayoutOptimizer():
    """
    LayoutOptimizer is a base class for wind plant layout optimization,
    acting as a wrapper for pyOptSparse.

    """

    def _base_init_(self, layout_x, layout_y, boundaries, solver="SNOPT", timer=None, options=None, history_file='hist.hist', output_file='out.out'):

        # Save boundary information
        self._boundaries = boundaries
        self._boundary_polygon = Polygon(boundaries)
        self._boundary_line = LineString(boundaries)

        # Position normalization
        self._xmin = np.min([tup[0] for tup in boundaries])
        self._xmax = np.max([tup[0] for tup in boundaries])
        self._ymin = np.min([tup[1] for tup in boundaries])
        self._ymax = np.max([tup[1] for tup in boundaries])

        self._x0 = self._norm(layout_x, self._xmin, self._xmax)
        self._y0 = self._norm(layout_y, self._ymin, self._ymax)
        self._nturbs = len(layout_x)

        # Optimization initialization
        self.solver = solver
        self.storeHistory = history_file
        self.timeLimit = timer
        self.optProb = pyoptsparse.Optimization('layout', self._obj_func)

        self.optProb = self.add_var_group(self.optProb)
        self.optProb = self.add_con_group(self.optProb)
        self.optProb.addObj("obj")

        # Optimizer options
        if options is not None:
            self.optOptions = options
        elif solver == "SNOPT":
            self.optOptions = {
                "Print file": output_file,
                "Major optimality tolerance": 1e-4,
                "Major feasibility tolerance": 1e-4,
                "Scale option": 2,
                }
        elif solver == "SLSQP":
            self.optOptions = {
                "ACC": 1e-6,
                "IFILE": output_file,
                "MAXIT": 10,
            }
        elif solver == "NSGA2":
            self.optOptions = {
                "maxGen": 10,
            }

        exec("self.opt = pyoptsparse." + self.solver + "(options=self.optOptions)")
    
    def _norm(self, val, x1, x2):
        """Method to normalize turbine positions"""
        return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        """Method to dimensionalize turbine positions"""
        return np.array(val) * (x2 - x1) + x1
    
    def optimize(self):
        """Method to initiate optimization."""
        self._optimize()
        return self.sol

    ###########################################################################
    # User constraint function
    ###########################################################################
    # TODO: fill in analytic boundary constraint
    def _boundary_constraint(self):
        boundary_con = np.zeros(self._nturbs)
        for i in range(self._nturbs):
            loc = Point(self._x[i], self._y[i])
            boundary_con[i] = loc.distance(self._boundary_line)
            if self._boundary_polygon.contains(loc)==True:
                boundary_con[i] *= -1.0

        return boundary_con

    ###########################################################################
    # pyOptSparse wrapper functions
    ###########################################################################
    def _optimize(self):
        if hasattr(self, "_sens_func"):
            if self.timeLimit is not None:
                self.sol = self.opt(self.optProb, storeHistory=self.storeHistory, sens=self._sens_func, timeLimit=self.timeLimit)
            else:
                self.sol = self.opt(self.optProb, storeHistory=self.storeHistory, sens=self._sens_func)
        else:
            if self.timeLimit is not None:
                self.sol = self.opt(self.optProb, storeHistory=self.storeHistory, timeLimit=self.timeLimit)
            else:
                self.sol = self.opt(self.optProb, storeHistory=self.storeHistory, sens='FD')

    def parse_opt_vars(self, varDict):
        self._x = self._unnorm(varDict["x"], self._xmin, self._xmax)
        self._y = self._unnorm(varDict["y"], self._ymin, self._ymax)

    def parse_sol_vars(self, sol):
        return np.array(self._unnorm(sol.getDVs()["x"], self._xmin, self._xmax)), np.array(self._unnorm(sol.getDVs()["y"], self._ymin, self._ymax))

    def parse_hist_vars(self, hist):
        val = hist.getValues(names=['x','y'],major=True)
        return np.array(self._unnorm(val['x'], self._xmin, self._xmax)), np.array(self._unnorm(val['y'], self._ymin, self._ymax))

    def add_var_group(self, optProb):
        optProb.addVarGroup("x", self._nturbs, type="c", lower=0.0, upper=1.0, value=self._x0)
        optProb.addVarGroup("y", self._nturbs, type="c", lower=0.0, upper=1.0, value=self._y0)
        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup("boundary_con", self._nturbs, upper=0.0)
        return optProb
    
    def compute_cons(self, funcs):
        funcs["boundary_con"] = self._boundary_constraint()
        return funcs


class FlowersOptimizer(LayoutOptimizer):
    """
    Child class of LayoutOptimizer for the FLOWERS-based layout optimizer.

    Args:
        flowers_interface (FlowersInterface): FLOWERS interface for calculating AEP
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        boundaries (list(tuple(float, float))): (x,y) position of each boundary point [m]
        solver (str, optional): the optimization algorithm to use:
            - "SLSQP" (default)
            - "SNOPT"
        timer (int, optional): time limit [s]
        history (str, optional): file name for pyoptsparse history file
        output (str, optional): file name for solver output file

    """

    def __init__(self, flowers_interface, layout_x, layout_y, boundaries, solver="SNOPT", timer=None, history_file='hist.hist', output_file='out.out'):
        self.model = flowers_interface
        self._aep_initial = self.model.calculate_aep()
        self._base_init_(layout_x, layout_y, boundaries, solver=solver, timer=timer, history_file=history_file, output_file=output_file)

    def _obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        self.model.reinitialize(layout_x=self._x, layout_y=self._y)

        # Compute the objective function
        funcs = {}
        funcs["obj"] = (
            -1 * self.model.calculate_aep() / self._aep_initial
        )

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    # Optionally, the user can supply the optimization with gradients
    # def _sens_func(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail


class ConventionalOptimizer(LayoutOptimizer):
    """
    Child class of LayoutOptimizer for the FLORIS-based layout optimizer.

    Args:
        floris_interface (FlorisInterface): FLORIS interface for calculating AEP
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        boundaries (list(tuple(float, float))): (x,y) position of each boundary point [m]
        solver (str, optional): the optimization algorithm to use:
            - "SLSQP" (default)
            - "SNOPT"
        timer (int, optional): time limit [s]
        history (str, optional): file name for pyoptsparse history file
        output (str, optional): file name for solver output file

    """

    def __init__(self, floris_interface, freq_val, layout_x, layout_y, boundaries, solver="SNOPT", timer=None, history_file='hist.hist', output_file='out.out'):
        self.model = floris_interface
        self._freq_1D = freq_val

        self.model.calculate_wake()
        self._aep_initial = np.sum(self.model.get_farm_power() * self._freq_1D * 8760)
        self._base_init_(layout_x, layout_y, boundaries, solver=solver, timer=timer, history_file=history_file, output_file=output_file)

    def _obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        self.model.reinitialize(layout_x=self._x.flatten(), layout_y=self._y.flatten())

        # Compute the objective function
        self.model.calculate_wake()
        funcs = {}
        funcs["obj"] = (
            -1 * np.sum(self.model.get_farm_power() * self._freq_1D * 8760) / self._aep_initial
        )

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    # Optionally, the user can supply the optimization with gradients
    # def _sens_func(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail