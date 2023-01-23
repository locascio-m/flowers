import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

import visualization as vis

wd = np.arange(0., 360., 1.)
ws = 10.
a = 1
c = 10
freq = a*(np.exp(-(wd-270)**2/(2*c)**2))
freq /= np.sum(freq)

df = pd.DataFrame()
df["wd"] = wd
df["ws"] = ws
df["freq_val"] = freq

vis.plot_wind_rose(df)

plt.show(block=False)

var = input("Would you like to save this wind rose? [y/n]: ")

if var == 'y':
    idx = input("Provide an index for this wind rose: ")
    file_name = './wind_roses/wr' + idx + '.p'
    df.to_pickle(file_name)

plt.close()