import model_interface as inter
import tools as tl
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wr = tl.load_wind_rose(8)
layout_x, layout_y, boundaries = tl.load_layout(0,"small")

# wr = tl.load_wind_rose(1)
# layout_x, layout_y, boundaries = tl.load_layout(0,"medium")

# wr = tl.load_wind_rose(6)
# layout_x, layout_y, boundaries = tl.load_layout(0,"large")

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)
solution = opt.run_optimization(optimizer="flowers", solver="SNOPT", gradient="numerical", scale=1e3, tol=1e-2, timer=60)

print(solution["init_aep"])
print(solution["opt_aep"])
print(solution["total_time"])
print(solution["iter"])
print(solution["obj_calls"])
print("Exit code: " + str(solution["exit_code"]))

vis.plot_optimal_layout(np.array(boundaries), solution["opt_x"],solution["opt_y"],solution["init_x"],solution["init_y"])
vis.plot_wind_rose(wr)

plt.figure()
plt.plot(range(solution["iter"]),solution["hist_aep"]/1e9)
plt.xlabel('Iteration')
plt.ylabel('AEP [GWh]')

# fig, ax = plt.subplots(1,1)
# ax.set(aspect='equal', xlim=[-5,25], ylim=[-5,25], xlabel='x/D', ylabel='y/D')
# # line, = ax.plot(solution["hist_x"][0]/126.,solution["hist_y"][0]/126.,"o",color='tab:red',markersize=7)
# patches = []
# for n in range(len(solution["hist_x"][0])):
#     patches.append(ax.add_patch(plt.Circle((solution["hist_x"][0,n], solution["hist_y"][0,n]), 1/2, color='tab:red')))

# def animate(i):
#     for n in range(len(solution["hist_x"][0])):
#         patches[n].center = solution["hist_x"][i,n]/126., solution["hist_y"][i,n]/126.
#     # line.set_data(solution["hist_x"][i]/126., solution["hist_y"][i]/126.)
#     ax.set(title=str(i))

# mov = anim.FuncAnimation(fig, animate, frames=solution["iter"], repeat=True)

# mov.save("./figures/opt_example.gif", dpi=500)

plt.show()

