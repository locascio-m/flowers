import model_interface as inter
import tools as tl
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wr = tl.load_wind_rose(6)
layout_x = 126. * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.])
boundaries = [(0., 0.),(0, 18*126.),(18*126, 18*126.),(18*126, 0.)]
opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)
solution = opt.run_optimization(optimizer="flowers", solver="SNOPT")

print(solution["init_aep"])
print(solution["opt_aep"])

vis.plot_optimal_layout(np.array(boundaries), solution["opt_x"],solution["opt_y"],solution["init_x"],solution["init_y"])

# plt.figure()
# plt.plot(range(solution["iter"]),solution["hist_aep"])

# fig, ax = plt.subplots(1,1)
# ax.set(aspect='equal', xlim=[-5,20], ylim=[-5,20], xlabel='x/D', ylabel='y/D')
# line, = ax.plot([],[],"o",color='tab:red',markersize=12)

# def animate(i):
#     line.set_data(solution["hist_x"][i]/126., solution["hist_y"][i]/126.)

# mov = anim.FuncAnimation(fig, animate, frames=solution["iter"], repeat=True)

plt.show()

