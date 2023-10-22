import model_interface as inter
import numpy as np
import tools as tl
import pickle

idx = 16
model = "flowers"
gradients = "analytical"
# k = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
k = 0.10

wr = tl.load_wind_rose(1)
boundaries = [(3*126., 0.),(12*126, 0.),(15*126, 10*126.),(10*126, 15*126.),(0, 15*126.)]
file_base = 'solutions/opt_k'

save_file = file_base + str(k)[-2:] + '.p'
history_file = file_base + str(k)[-2:] + '.hist'
output_file = file_base + str(k)[-2:] + '.out'

layout_x, layout_y = tl.random_layout(boundaries, n_turb=20, idx=idx)
opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries, k=k)

solution = opt.run_optimization(optimizer=model, solver="SNOPT", gradient=gradients, timer=300, history=history_file, output=output_file)

boundaries = np.array(boundaries).T
pickle.dump((solution, wr, boundaries), open(save_file,'wb'))