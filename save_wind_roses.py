# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt

import tools as tl
import visualization as vis

"""
This file downloads a wind rose from the WIND Toolkit and gives the 
user the option to save it to a file '.wind_rose/wr#.p', where # is a 
user-provided index.

"""

wind_rose = tl.toolkit_wind_rose(lat = 41.05, long = -70.65)

vis.plot_wind_rose(wind_rose)

plt.show(block=False)

var = input("Would you like to save this wind rose? [y/n]: ")

if var == 'y':
    idx = input("Provide an index for this wind rose: ")
    file_name = './wind_roses/wr' + idx + '.p'
    wind_rose.to_pickle(file_name)

plt.close()