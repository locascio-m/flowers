# FLOWERS

# Michael LoCascio

import numpy as np
import model as set
import tools as tl
import matplotlib.pyplot as plt
import visualization as vis

"""
This file is a workspace for testing the vectorization of the FLOWERS
code and the implementation of automatic differentiation.
"""


layout_x = 126. * np.array([0.0])
layout_y = 126. * np.array([0.0])
wind_rose = tl.load_wind_rose(2)

geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
geo.compare_aep(num_terms=181, wd_resolution=1.0, ws_avg=True)
plt.show()