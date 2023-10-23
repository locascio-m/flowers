import model_interface as inter
import tools as tl
import numpy as np
import matplotlib.pyplot as plt

wr = tl.load_wind_rose(7)
layout_x = 126. * np.array([0.])
layout_y = 126. * np.array([0.])
model = inter.AEPInterface(wr, layout_x, layout_y)
model.reinitialize(ws_avg=True)
model.compare_aep()
