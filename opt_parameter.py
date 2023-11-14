import model_interface as inter
import numpy as np
import tools as tl
import pickle
import sys

idx = int(sys.argv[3])
farm = str(sys.argv[1])
k = float(sys.argv[2])

tol = 1e-3
if farm == "small":
    wr = tl.load_wind_rose(8)
    scale = 1e2
elif farm == "medium":
    wr = tl.load_wind_rose(1)
    scale = 1e2
elif farm == "large":
    wr = tl.load_wind_rose(6)
    scale = 1e3

layout_x, layout_y, boundaries = tl.load_layout(idx, farm)

file_base = 'solutions/opt_' + farm + '_parameter_' + '{:.2f}'.format(k) + '_'

save_file = file_base + str(idx) + '.p'
# history_file = file_base + str(idx) + '.hist'
output_file = file_base + str(idx) + '.out'

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries, k=k)

solution = opt.run_optimization(optimizer="flowers", solver="SNOPT", gradient="analytical", timer=86400, output=output_file)

boundaries = np.array(boundaries).T
pickle.dump((solution, wr, boundaries), open(save_file,'wb'))