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
        self._boundaries = np.array(boundaries).T
        self._nbounds = len(self._boundaries[0])

        # Compute edge information
        self._boundary_edge = np.roll(self._boundaries,-1,axis=1) - self._boundaries
        self._boundary_len = np.sqrt(self._boundary_edge[0]**2 + self._boundary_edge[1]**2)
        self._boundary_norm = np.array([self._boundary_edge[1],-self._boundary_edge[0]]) / self._boundary_len
        self._boundary_int = (np.roll(self._boundary_norm,1,axis=1) + self._boundary_norm) / 2

        # Position normalization
        self._xmin = np.min(self._boundaries[0])
        self._xmax = np.max(self._boundaries[0])
        self._ymin = np.min(self._boundaries[1])
        self._ymax = np.max(self._boundaries[1])

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
                "iSumm": 0,
                "Major optimality tolerance": 1e-4,
                "Minor optimality tolerance": 1e-4,
                "Major feasibility tolerance": 1e-4,
                "Minor feasibility tolerance": 1e-4,
                "Scale option": 0,
                }
        elif solver == "SLSQP":
            self.optOptions = {
                "ACC": 1e-6,
                "IFILE": output_file,
                "MAXIT": 100,
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
    def _boundary_constraint(self, gradient=False):
        # Transform inputs
        points = np.array([self._x,self._y])

        # Compute distances from turbines to boundary points
        a = np.zeros((self._nturbs,2,self._nbounds))
        for i in range(self._nturbs):
            a[i] = np.expand_dims(points[:,i].T,axis=-1) - self._boundaries

        # Compute projections
        a_edge = np.sum(a*self._boundary_edge, axis=1) / self._boundary_len
        a_int = np.sum(a*self._boundary_norm, axis=1)
        sigma = np.sign(np.sum(a*self._boundary_int, axis=1))

        # Initialize signed distance containers
        C = np.zeros(self._nturbs)
        D = np.zeros(self._nbounds)
        if gradient:
            Cx = np.zeros(self._nturbs)
            Cy = np.zeros(self._nturbs)

        # Compute signed distance
        for i in range(self._nturbs):
            for k in range(self._nbounds):
                if a_edge[i,k] < 0:
                    D[k] = np.sqrt(a[i,0,k]**2 + a[i,1,k]**2)*sigma[i,k]
                elif a_edge[i,k] > self._boundary_len[k]:
                    D[k] = np.sqrt(a[i,0,(k+1)%self._nbounds]**2 + a[i,1,(k+1)%self._nbounds]**2)*sigma[i,(k+1)%self._nbounds]
                else:
                    D[k] = a_int[i,k]
            
            # Select minimum distance
            idx = np.argmin(np.abs(D))
            C[i] = D[idx]

            if gradient:
                if a_edge[i,idx] < 0:
                    Cx[i] = (points[0,i] - self._boundaries[0,idx]) / np.sqrt((self._boundaries[0,idx]-points[0,i])**2 + (self._boundaries[1,idx]-points[1,i])**2)
                    Cy[i] = (points[1,i] - self._boundaries[1,idx]) / np.sqrt((self._boundaries[0,idx]-points[0,i])**2 + (self._boundaries[1,idx]-points[1,i])**2)
                elif a_edge[i,idx] > self._boundary_len[idx]:
                    Cx[i] = (points[0,i] - self._boundaries[0,(idx+1)%self._nbounds]) / np.sqrt((self._boundaries[0,(idx+1)%self._nbounds]-points[0,i])**2 + (self._boundaries[1,(idx+1)%self._nbounds]-points[1,i])**2)
                    Cy[i] = (points[1,i] - self._boundaries[1,(idx+1)%self._nbounds]) / np.sqrt((self._boundaries[0,(idx+1)%self._nbounds]-points[0,i])**2 + (self._boundaries[1,(idx+1)%self._nbounds]-points[1,i])**2)
                else:
                    Cx[i] = (self._boundaries[1,(idx+1)%self._nbounds] - self._boundaries[1,idx]) / self._boundary_len[idx]
                    Cy[i] = (self._boundaries[0,idx] - self._boundaries[0,(idx+1)%self._nbounds]) / self._boundary_len[idx]
        
        # Distance is negative if inside boundary for optimization problem 
        if gradient:
            return self._norm(C, self._xmin, self._xmax), Cx, Cy
        else:
            return self._norm(C, self._xmin, self._xmax)

    ###########################################################################
    # pyOptSparse wrapper functions
    ###########################################################################
    def _optimize(self):
        if self.gradient:
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
        optProb.addVarGroup("x", self._nturbs, type="c", value=self._x0)
        optProb.addVarGroup("y", self._nturbs, type="c", value=self._y0)
        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup("con", self._nturbs, upper=0.0)
        return optProb
    
    def compute_cons(self, funcs):
        funcs["con"] = self._boundary_constraint()
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

    def __init__(self, flowers_interface, layout_x, layout_y, boundaries, grad="analytical", solver="SNOPT", timer=None, history_file='hist.hist', output_file='out.out'):
        self.model = flowers_interface
        if grad == "analytical":
            self.gradient = True
        elif grad == "numerical":
            self.gradient = False
        self._aep_initial, self._grad_initial = self.model.calculate_aep(gradient=True)
        self._grad_initial = 1e-5*self._aep_initial
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
    def _sens_func(self, varDict, funcs):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        self.model.reinitialize(layout_x=self._x, layout_y=self._y)

        _, tmp = self.model.calculate_aep(gradient=True)
        funcsSens = {}
        funcsSens["obj"] = {"x": -tmp[:,0]/self._grad_initial, "y": -tmp[:,1]/self._grad_initial}
        
        _, tmpx, tmpy = self._boundary_constraint(gradient=True)
        funcsSens["con"] = {"x": np.diag(tmpx), "y": np.diag(tmpy)}

        fail = False

        return funcsSens, fail


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

    def __init__(self, floris_interface, freq_val, layout_x, layout_y, boundaries, grad="analytical", solver="SNOPT", timer=None, history_file='hist.hist', output_file='out.out'):
        self.model = floris_interface
        self.gradient = False
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