import model_interface as inter
import tools as tl
import numpy as np
import matplotlib.pyplot as plt
import visualization as viz

wr = tl.load_wind_rose(7)
# layout_x = 126. * np.array([0.])
# layout_y = 126. * np.array([0.])
layout_x, layout_y = tl.discrete_layout(10)
viz.plot_layout(layout_x,layout_y)
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

model = inter.AEPInterface(wr, layout_x, layout_y, conventional_model='jensen')
model.reinitialize(ws_avg=True)
model.compare_aep()
