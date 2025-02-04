
import numpy as np
import pandas as pd

from flowers import FlowersModel


# Create generic layout
D = 126.
layout_x = D * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.,21.,21.,21.,28.,28.,28.])
layout_y = D * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.,0.,7.,14.,0.,7.,14.])

# Load in wind data
df = pd.read_csv('inputs/HKW_wind_rose.csv')

# Setup FLOWERS model
flowers_model = FlowersModel(df, layout_x, layout_y)

# Calculate the AEP
aep = flowers_model.calculate_aep()
print(aep)
