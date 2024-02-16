import model_interface as inter
import tools as tl
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pickle

model = "flowers"
# model = "gauss"
# model = "jensen"

if model == "flowers":
    optimizer = "flowers"
    gradient = "analytical"
    scale = 1e4
    conventional = None
else:
    optimizer = "conventional"
    gradient = "numerical"
    scale = 1e4
    conventional = model

save_file = 'solutions/aep_layout_' + model + '.p'

# Load layout, boundaries, and rose
layout_x = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])
boundaries = [(14*126, 0.),(14*126, 14*126.),(0, 14*126.),(0., 0.)]
wr = tl.load_wind_rose(7)

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries, conventional_model=conventional)
solution = opt.run_optimization(optimizer=optimizer, solver="SNOPT", gradient=gradient, scale=scale, tol=1e-3, timer=600)

# print(solution["init_aep"])
# print(solution["opt_aep"])
# print(solution["total_time"])
# print(solution["iter"])
# print("Exit code: " + str(solution["exit_code"]))

# vis.plot_optimal_layout(np.array(boundaries), solution["opt_x"],solution["opt_y"],solution["init_x"],solution["init_y"])
# vis.plot_wind_rose(wr)

boundaries = np.array(boundaries).T

pickle.dump((solution, wr, boundaries), open(save_file,'wb'))

