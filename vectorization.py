# FLOWERS

# Michael LoCascio

import numpy as np
import flowers_interface as flow
import tools as tl
import time

# Wind rose (sampled from stored wind roses)
wind_rose = tl.load_wind_rose(1)

# Number of turbines
n_turb = 31

# Wind rose resolution
num_terms = 181

# Randomize wind farm layout
xx = np.linspace(0., 6300., 10)
yy = np.linspace(0., 6300., 10)
layout_x, layout_y = np.meshgrid(xx,yy)
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()

# Initialize optimization interface

fi = flow.Flowers(wind_rose,layout_x,layout_y,k=0.05,D=126.0)
fi.fourier_coefficients(num_terms=num_terms)
t = time.time()
aep = fi.calculate_aep()
elapsed = time.time() - t
print(aep)
print(elapsed)