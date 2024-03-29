# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import os
import pandas as pd

import visualization as vis

"""
This file visualizes all of the wind roses saved in the
./wind_rose/ directory.

"""

for file in os.listdir('./wind_roses/'):
    if file[-1] == 'p':
        df = pd.read_pickle('./wind_roses/' + file)
        vis.plot_wind_rose(df)
        ax = plt.gca()
        ax.set(title=file)
plt.show()