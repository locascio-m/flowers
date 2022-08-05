# FLOWERS

# Michael LoCascio

import matplotlib.pyplot as plt
import numpy as np
import pickle

import tools as tl
import visualization as vis

"""
This file collects the individual randomized cases of the multistart study
and computes average data: optimal AEP, solver time, and optimal layouts.

The ModelComparison objects are real from 'multi#.p' files generated by the
multistart.py script.
"""

# Number of random cases
multi = 50
flowers_flag = True
floris_flag = False

# Initialize collected data
aep_flowers = np.zeros(multi)
aep_floris = np.zeros(multi)
time_flowers = np.zeros(multi)
time_floris = np.zeros(multi)

# Format plots
font = 14
plt.rc('font', size=font)          # controls default text sizes
plt.rc('axes', titlesize=font)     # fontsize of the axes title
plt.rc('axes', labelsize=font)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font)    # fontsize of the tick labels
plt.rc('legend', fontsize=font-2)    # legend fontsize
plt.rc('figure', titlesize=font)  # fontsize of the figure title

# Superimposed layouts
_, (ax0, ax00) = plt.subplots(1,2, figsize=(12,4.75))

# AEP
_, (ax1, ax11) = plt.subplots(1,2, figsize=(12,4.75))

# Wind rose
fig = plt.figure(figsize=(6,4.75))
ax3 = fig.add_subplot(projection='polar')

# Plot generic initial layout
layout_x, layout_y = tl.load_layout('iea')
vis.plot_layout(layout_x, layout_y)

for i in range(multi):

    # Read data from each case
    if flowers_flag:
        file_name = 'solutions/flowers_' + str(i) + '.p'
        sol = pickle.load(open(file_name,'rb'))
        time_flowers[i] = sol.flowers_solution['time']
        aep_flowers[i] = sol.aep_flowers
        ax0.plot(sol.layout_flowers[0]/sol.diameter, sol.layout_flowers[1]/sol.diameter, "o", markersize=6, color='#21918c', alpha=0.5)
    if floris_flag:
        file_name = 'solutions/floris' + str(i) + '.p'
        sol = pickle.load(open(file_name,'rb'))
        time_floris[i] = sol.floris_solution['time']
        aep_floris[i] = sol.aep_floris
        ax00.plot(sol.layout_floris[0]/sol.diameter, sol.layout_floris[1]/sol.diameter, "o", markersize=6, color='#21918c', alpha=0.5)

# Plot optimal AEP and solver time
if flowers_flag:
    ax1.plot(time_flowers, aep_flowers/1e9, 'o', color='#440154')
    ax1.set(xlabel="Time (s)", ylabel="AEP (GWh)", xlim=0, title='FLOWERS')
    ax1.grid(True)
if floris_flag:
    ax11.plot(time_floris, aep_floris/1e9, 'o', color='#440154')
    ax11.set(xlabel="Time (s)", ylabel="AEP (GWh)", xlim=0, title="FLORIS")
    ax11.grid(True)

# Plot wind farm boundary
verts = sol.boundaries/sol.diameter
for i in range(len(verts)):
    if i == len(verts) - 1:
        ax0.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
        ax00.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
    else:
        ax0.plot(
            [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "black"
        )
        ax00.plot(
            [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "black"
        )

ax0.set(xlabel="x / D", ylabel="y / D", title='FLOWERS', aspect='equal')
ax00.set(xlabel="x / D", ylabel="y / D", title='FLORIS', aspect='equal')
ax0.grid()
ax00.grid()

# Plot wind rose
vis.plot_wind_rose(sol.wind_rose, ax=ax3)

# Output averaged results
print("================================")
print('Multistart Results    ')
print('Number of Cases:              {:.0f}'.format(multi))
print('FLOWERS Terms:                {:.0f}'.format(sol.terms_flowers))
print('FLORIS Bins:                  {:.0f}'.format(sol.bins_floris))
print()
print('FLOWERS AEP Mean:    {:.3f} GWh'.format(np.mean(aep_flowers) / 1e9))
print('FLOWERS AEP Std:       {:.3f} GWh'.format(np.std(aep_flowers) / 1e9))
print('FLORIS AEP Mean:     {:.3f} GWh'.format(np.mean(aep_floris) / 1e9))
print('FLORIS AEP Std:       {:.3f} GWh'.format(np.std(aep_floris) / 1e9))
print()
print('FLOWERS Time Mean:    {:.2f} s'.format(np.mean(time_flowers)))
print('FLOWERS Time Std:     {:.2f} s'.format(np.std(time_flowers)))
print('FLORIS Time Mean:     {:.2f} s'.format(np.mean(time_floris)))
print('FLORIS Time Std:      {:.2f} s'.format(np.std(time_floris)))
# print('Conv. Time Mean:     {:.2f} s'.format(np.mean(np.sort(time)[:-8])))
# print('Conv. Time Std:      {:.2f} s'.format(np.std(np.sort(time)[:-8])))
# print()
# print('Solution A AEP:  {:.3f} GWh'.format(aep[19] / 1e9))
# print('Solution B AEP:  {:.3f} GWh'.format(aep[71] / 1e9))
# print('Solution A Time:     {:.2f} s'.format(time[19]))
# print('Solution B Time:     {:.2f} s'.format(time[71]))
print("================================")
plt.show()