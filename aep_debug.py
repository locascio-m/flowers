import model_interface as inter
import tools as tl
import floris
import numpy as np

wr = tl.load_wind_rose(6)
# layout_x, layout_y = tl.discrete_layout(24, idx=12)
layout_x = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])
model = inter.AEPInterface(wr, layout_x, layout_y, num_terms=0, conventional_model='gauss')
model.reinitialize(wd_resolution=5.0,ws_avg=False)
model.compare_aep()

