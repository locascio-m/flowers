import pickle
import tools

farm = "small"
# farm = "medium"
# farm = "large"

if farm == "small":
    boundaries = [(0., 0.),(10*126, 0.),(10*126, 10*126.),(0., 10*126.)]
    nt = 10
elif farm == "medium":
    boundaries = [(8*126., 0.),(28*126, 0.),(36*126, 24*126.),(24*126, 36*126.),(0, 36*126.)]
    nt = 50
elif farm == "large":
    boundaries = [(10.*126., 0.),(125*126., 0.),(125*126., 50*126.),(110*126., 160*126.),(40*126., 160*126.),(40*126., 120*126.),(0*126., 100*126.),(12.*126., 40*126.),(15.*126., 20.*126.)]
    nt = 250

for idx in range(100):
    file = './layouts/' + farm + str(idx) + '.p'
    xx, yy = tools.random_layout(boundaries, n_turb=nt, idx=idx)
    pickle.dump((xx, yy, boundaries), open(file,'wb'))