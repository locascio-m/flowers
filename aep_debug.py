import model_interface as inter
import tools as tl
import floris
import numpy as np

wr = tl.load_wind_rose(6)
layout_x = 126. * np.array([0.,0.,-3.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,-2.,7.,14.])
boundaries = [(0., 0.),(18*126, 0.),(18*126, 18*126.),(0, 10*126.)]
model = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)
print(model._aep_initial)
print(np.where(np.isnan(model.post_processing.get_farm_power())))

