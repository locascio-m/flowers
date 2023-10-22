import model_interface as inter
import numpy as np
import tools as tl
import pickle

idx = 0
model = "conventional"
gradients = "numerical"

wr = tl.load_wind_rose(1)
boundaries = [(3*126., 0.),(12*126, 0.),(15*126, 10*126.),(10*126, 15*126.),(0, 15*126.)]
file_base = 'solutions/opt_floris'

save_file = file_base + str(idx) + '.p'
history_file = file_base + str(idx) + '.hist'
output_file = file_base + str(idx) + '.out'

layout_x, layout_y = tl.random_layout(boundaries, n_turb=20, idx=idx)
opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)

solution = opt.run_optimization(optimizer=model, solver="SNOPT", gradient=gradients, timer=600, history=history_file, output=output_file)

boundaries = np.array(boundaries).T

print(solution['init_aep'])
print(solution['opt_aep'])
print(solution['total_time'])
print(solution['iter'])
# pickle.dump((solution, wr, boundaries), open(save_file,'wb'))