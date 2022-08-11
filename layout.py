# FLOWERS

# Michael LoCascio

import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, LineString


def _norm(val, x1, x2):
        return (val - x1) / (x2 - x1)

def _unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1


class LayoutOptimization:
    """
    LayoutOptimization is a high-level interface to set up the layout optimization
    problem for FLOWERS. AEP is the objective function. Two constraints are enforced:
    (1) minimum spacing of 2 rotor diameters and (2) all turbines must be within the
    farm boundary. This code is copied from FLORIS except for obj_func()

    Args:
        fi (Flowers): FLOWERS interface
        boundaries (list(tuple)): boundary vertices in the form
                [(x0,y0), (x1,y1), ... , (xN,yN)]
        scale_dv (float, optional): scaling for design variables
            (turbine positions). Defaults to 1.0
        scale_con (float, optional): scaling for constraints
            (spacing and boundary). Defaults to 1.0

    """
    
    def __init__(self, fi, boundaries, min_dist=2.0):

        # Import FLOWERS interface
        self.fi = fi

        # Import boundary
        self.boundaries = boundaries
        self.boundary_polygon = Polygon(self.boundaries)
        self.boundary_line = LineString(self.boundaries)

        # Position normalization
        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])

        self.x0 = _norm(self.fi.layout_x, self.xmin, self.xmax)
        self.y0 = _norm(self.fi.layout_y, self.ymin, self.ymax)

        # Minimum separation
        self.min_dist = min_dist * self.fi.D

        # Optimization initialization

        # Compute initial AEP
        self.initial_AEP = fi.calculate_aep()
    
    def __str__(self):
        return "layout"
    
    def reinitialize(self):
        pass

    def obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        self.fi.layout_x = self.x
        self.fi.layout_y = self.y

        # Compute the objective function
        funcs = {}
        funcs["obj"] = (
            -1 * self.fi.calculate_aep() / self.initial_AEP
        )

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    def parse_opt_vars(self, varDict):
        self.x = _unnorm(varDict["x"], self.xmin, self.xmax)
        self.y = _unnorm(varDict["y"], self.ymin, self.ymax)

    def parse_sol_vars(self, sol):
        self.x = list(_unnorm(sol.getDVs()["x"], self.xmin, self.xmax))[0]
        self.y = list(_unnorm(sol.getDVs()["y"], self.ymin, self.ymax))[1]

    def add_var_group(self, optProb):
        optProb.addVarGroup(
            "x", self.nturbs, type="c", lower=0.0, upper=1.0, value=self.x0, scale=self.scale_dv
        )
        optProb.addVarGroup(
            "y", self.nturbs, type="c", lower=0.0, upper=1.0, value=self.y0, scale=self.scale_dv
        )

        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup("boundary_con", self.nturbs, upper=0.0, scale=self.scale_con)
        optProb.addConGroup("spacing_con", 1, upper=0.0, scale=self.scale_con)

        return optProb
    
    def compute_cons(self, funcs):
        funcs["boundary_con"] = self.distance_from_boundaries()
        funcs["spacing_con"] = self.space_constraint()

        return funcs
    
    def space_constraint(self, rho=500):
        x = self.x
        y = self.y

        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)

        g = 1 - np.array(dist) / self.min_dist

        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)

        return KS_constraint[0][0]

    def distance_from_boundaries(self):
        boundary_con = np.zeros(self.nturbs)
        for i in range(self.nturbs):
            loc = Point(self.x[i], self.y[i])
            boundary_con[i] = loc.distance(self.boundary_line) #NaNsafe, or 1 to 5 m inside boundary
            if self.boundary_polygon.contains(loc)==True:
                boundary_con[i] *= -1.0

        return boundary_con

    def analytic_gradients(self):
        return AnalyticGradients(self)

    def get_optimal_layout(self, sol):
        locsx = _unnorm(sol.getDVs()["x"], self.xmin, self.xmax)
        locsy = _unnorm(sol.getDVs()["y"], self.ymin, self.ymax)

        return locsx, locsy

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = len(self.fi.layout_x)
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.D

class AnalyticGradients:
    # positions: dictionary of design variable values
    # values: dictionary of objective and constraint values
    # returns: nested dictionary of gradient with respect to each variable for objective and constraints
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, positions, values):
        return None