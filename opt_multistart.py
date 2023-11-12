import model_interface as inter
import numpy as np
import sys
import tools as tl
import pickle

idx = int(sys.argv[4])
farm = str(sys.argv[3])
model = str(sys.argv[1])
gradients = str(sys.argv[2])

if farm == "small":
    wr = tl.load_wind_rose(8)
    if model == "flowers":
        scale = 1e2
    elif model == "conventional":
        scale = 1e3
    tol = 1e-3
elif farm == "medium":
    wr = tl.load_wind_rose(1)
    if model == "flowers":
        scale = 1e2
    elif model == "conventional":
        scale = 1e3
    tol = 1e-3
elif farm == "large":
    wr = tl.load_wind_rose(6)
    scale = 1e3
    tol = 1e-2

layout_x, layout_y, boundaries = tl.load_layout(idx, farm)

file_base = 'solutions/opt_' + farm + '_' + model + '_' + gradients + '_'

save_file = file_base + str(idx) + '.p'
history_file = file_base + str(idx) + '.hist'
output_file = file_base + str(idx) + '.out'

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)

solution = opt.run_optimization(optimizer=model, solver="SNOPT", gradient=gradients, scale=scale, tol=tol, timer=86400, history=history_file, output=output_file)

boundaries = np.array(boundaries).T

pickle.dump((solution, wr, boundaries), open(save_file,'wb'))