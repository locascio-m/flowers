import model_interface as inter
import tools as tl
import numpy as np
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.animation as anim

wr = tl.load_wind_rose(8)
# boundaries = [(0.,0.,),(30*126.,0.,),(30*126.,30*126.,),(0.,10*126.,)]
# layout_x, layout_y = tl.random_layout(boundaries=boundaries, n_turb=25,idx=4)
layout_x = 126. * np.array([0.,0.,-3.,7.,7.,7.,14.,14.,14.])
layout_y = 126. * np.array([0.,7.,14.,0.,7.,14.,-2.,7.,14.])
boundaries = [(0., 0.),(12*126, 0.),(12*126, 6*126.),(6*126, 12*126.),(0, 12*126.)]
# layout_x = 126. * np.array([0.,0.,1.,7.,7.,7.,14.,14.,14.])
# layout_y = 126. * np.array([0.,7.,9.,0.,7.,10.,0.,7.,14.])
# boundaries = [(0., 0.),(18*126, 0.),(18*126, 18*126.),(0, 10*126.)]
# layout_x = 126. * np.array([-2.,-2.,-2.,5.,5.,5.,12.,12.,12.])
# layout_y = 126. * np.array([-2.,5.,12.,-2.,5.,12.,-2.,5.,12.])
# boundaries = [(0., 0.),(10*126, 0.),(10*126, 10*126.),(0, 10*126.)]

opt = inter.WPLOInterface(wr, layout_x, layout_y, boundaries)
solution = opt.run_optimization(optimizer="flowers", solver="SNOPT", timer=10)

print(solution["init_aep"])
print(solution["opt_aep"])
print(solution["total_time"])
print(solution["iter"])
print(solution["obj_calls"])
print("Exit code: " + str(solution["exit_code"]))

vis.plot_optimal_layout(np.array(boundaries), solution["opt_x"],solution["opt_y"],solution["init_x"],solution["init_y"])
vis.plot_wind_rose(wr)

plt.figure()
plt.plot(range(solution["iter"]),solution["hist_aep"])

fig, ax = plt.subplots(1,1)
ax.set(aspect='equal', xlim=[-5,35], ylim=[-5,35], xlabel='x/D', ylabel='y/D')
line, = ax.plot(solution["hist_x"][0]/126.,solution["hist_y"][0]/126.,"o",color='tab:red',markersize=7)

def animate(i):
    line.set_data(solution["hist_x"][i]/126., solution["hist_y"][i]/126.)
    ax.set(title=str(i))

mov = anim.FuncAnimation(fig, animate, frames=solution["iter"], repeat=True)

# mov.save("./figures/opt_example.mp4")

plt.show()

