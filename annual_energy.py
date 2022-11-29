# FLOWERS

# Michael LoCascio

import numpy as np
import model as set
import tools as tl
import visualization as vis
import matplotlib.pyplot as plt
import warnings

"""
This file compares the AEP between FLOWERS and FLORIS for three different
wind roses and layouts. 

"""

warnings.filterwarnings("ignore")

# Overall parameters
D = 126.0
num_terms = 7
wd_resolution = 1.0
ws_avg = True

### Case 1: Small Aligned Wind Farm
print("Starting Case 1.")

# Define layout and wind rose
layout_x = D * np.array([0.0, 5., 10.])
layout_y = D * np.array([0.0, 0.0, 0.0])
wind_rose = tl.load_wind_rose(1)

fig1 = plt.figure(figsize=(12,4.75))
ax11 = fig1.add_subplot(121, polar=True)
ax12 = fig1.add_subplot(122)
vis.plot_wind_rose(wind_rose, ax=ax11)
vis.plot_layout(layout_x, layout_y, ax=ax12)

# Initialize and compute AEP
geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg)

print("Completing Case 1.")

### Case 2: Large Gridded Wind Farm
print("Starting Case 2.")

# Define layout and wind rose
xx = np.linspace(0., 70*D, 10)
layout_x, layout_y = np.meshgrid(xx,xx)
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()
wind_rose = tl.load_wind_rose(4)

fig2 = plt.figure(figsize=(12,4.75))
ax21 = fig2.add_subplot(121, polar=True)
ax22 = fig2.add_subplot(122)
vis.plot_wind_rose(wind_rose, ax=ax21)
vis.plot_layout(layout_x, layout_y, ax=ax22)

# Initialize and compute AEP
geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg)

print("Completing Case 2.")

### Case 3: Random Wind Farm
print("Starting Case 3.")

# Define layout and wind rose
boundaries = [(0.0, 0.0), (24*D, 0.0), (24*D, 24*D), (0.0, 24*D)]
layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=25, idx=21)
wind_rose = tl.load_wind_rose(6)

fig3 = plt.figure(figsize=(12,4.75))
ax31 = fig3.add_subplot(121, polar=True)
ax32 = fig3.add_subplot(122)
vis.plot_wind_rose(wind_rose, ax=ax31)
vis.plot_layout(layout_x, layout_y, ax=ax32)

# Initialize and compute AEP
geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
geo.compare_aep(num_terms=num_terms, wd_resolution=wd_resolution, ws_avg=ws_avg)

print("Completing Case 3.")

plt.show()