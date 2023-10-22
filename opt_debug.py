import model_interface as inter
import tools as tl
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wr = tl.load_wind_rose(8)
layout_x = 126. * np.array([0.,0.,-3.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,-2.,7.,14.])
boundaries = [(0., 0.),(12*126, 0.),(12*126, 6*126.),(6*126, 12*126.),(0, 12*126.)]

# wr = tl.load_wind_rose(1)
# layout_x = 126. * np.array([0.,0.,1.,7.,7.,7.,14.,14.,14.,18.,18.,18.])
# layout_y = 126. * np.array([0.,7.,9.,0.,7.,10.,0.,7.,14.,-1.,5.,10.])
# boundaries = [(0., 0.),(20*126, 0.),(20*126, 20*126.),(15*126, 20*126.),(15*126, 10*126.),(0, 10*126.)]

# layout_x = 126. * np.array([-2.,-2.,-2.,5.,5.,5.,12.,12.,12.])
# layout_y = 126. * np.array([-2.,5.,12.,-2.,5.,12.,-2.,5.,12.])
# boundaries = [(0., 0.),(10*126, 0.),(10*126, 10*126.),(0, 10*126.)]

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)
solution = opt.run_optimization(optimizer="conventional", solver="SNOPT", gradient="numerical", timer=60)

print(solution["init_aep"])
print(solution["opt_aep"])
print(solution["total_time"])
print(solution["iter"])
print(solution["obj_calls"])
print("Exit code: " + str(solution["exit_code"]))

vis.plot_optimal_layout(np.array(boundaries), solution["opt_x"],solution["opt_y"],solution["init_x"],solution["init_y"])
# plt.savefig('./figures/opt_example.png', dpi=500)
vis.plot_wind_rose(wr)

plt.figure()
plt.plot(range(solution["iter"]),solution["hist_aep"]/1e9)
plt.xlabel('Iteration')
plt.ylabel('AEP [GWh]')
# plt.savefig('./figures/opt_conv_example.png', dpi=500)

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

