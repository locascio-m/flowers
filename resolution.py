# FLOWERS

# Michael LoCascio

import numpy as np
import model as set
import tools as tl
import matplotlib.pyplot as plt
import visualization as vis

"""
This file is a workspace for testing the vectorization of the FLOWERS
code and the implementation of automatic differentiation.
"""

# # Format plots
# font = 14
# plt.rc('font', size=font)          # controls default text sizes
# plt.rc('axes', titlesize=font)     # fontsize of the axes title
# plt.rc('axes', labelsize=font)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=font)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=font)    # fontsize of the tick labels
# plt.rc('legend', fontsize=font-2)    # legend fontsize
# plt.rc('figure', titlesize=font)  # fontsize of the figure title

# # Wind rose (sampled from stored wind roses)
# wind_rose = tl.load_wind_rose(6)

# fig0 = plt.figure(figsize=(12,4.75))
# ax0 = fig0.add_subplot(121, polar=True)
# ax1 = fig0.add_subplot(122, polar=True)

# vis.plot_wind_rose(wind_rose, ax0)
# ax0.get_legend().remove()

# wr = tl.resample_average_ws_by_wd(wind_rose)

# wd = np.append(wr.wd, 360)
# wd = wd * np.pi / 180
# freq = wr.freq_val
# freq = np.append(freq, freq[0])
# ax1.plot(wd, freq, 'k', linewidth=2)
# ax1.set_theta_direction(-1)
# ax1.set_theta_offset(np.pi / 2.0)
# ax1.set_theta_zero_location("N")
# ax1.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# ax1.set_yticklabels([])

# f = np.fft.rfft(freq)
# f0 = np.fft.irfft(f*len(f))
# f1 = np.fft.irfft(f[:100]*len(f[:100]))
# f2 = np.fft.irfft(f[:25]*len(f[:25]))
# f3 = np.fft.irfft(f[:10]*len(f[:10]))
# f0[-1] = f0[0]
# f1[-1] = f1[0]
# f2[-1] = f2[0]
# f3[-1] = f3[0]

# fig1, ax = plt.subplots(subplot_kw=dict(polar=True))
# ax.plot(np.linspace(0, 2*np.pi, len(f0)), f0)
# ax.plot(np.linspace(0, 2*np.pi, len(f1)),f1)
# ax.plot(np.linspace(0, 2*np.pi, len(f2)),f2)
# ax.plot(np.linspace(0, 2*np.pi, len(f3)), f3)
# ax.set_theta_direction(-1)
# ax.set_theta_offset(np.pi / 2.0)
# ax.set_theta_zero_location("N")
# ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# ax.set_yticklabels([])
# ax.legend(
#     ['N = 181', 'N = 100', 'N = 25', 'N = 10'],
#     loc="lower left",
#     bbox_to_anchor=(.55 + np.cos(.55)/2, .5 + np.sin(.55)/2)
#     )

# fig0.tight_layout()
# fig1.tight_layout()


# Resolution test
xx = np.linspace(0., 35*126., 5)
layout_x, layout_y = np.meshgrid(xx,xx)
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()
bins = np.array([360, 180, 90, 72, 60, 45, 40, 36, 30, 24, 15, 12, 10, 8, 6, 5])

fig = plt.figure(figsize=(10,6))
ax3 = fig.add_subplot(223)
ax1 = fig.add_subplot(221, sharex=ax3)
ax4 = fig.add_subplot(224, sharey=ax3)
ax2 = fig.add_subplot(222, sharex=ax4, sharey=ax1)

for c in [1,3,6]:

    wind_rose = tl.load_wind_rose(c)
    geo = set.ModelComparison(wind_rose, layout_x, layout_y, model='gauss')
    flowers_terms = np.zeros(len(bins))
    flowers_aep = np.zeros(len(bins))
    flowers_time = np.zeros(len(bins))
    floris_bins = np.zeros(len(bins))
    floris_aep = np.zeros(len(bins))
    floris_time = np.zeros(len(bins))

    for i in range(len(bins)):
        wd = bins[i]
        N = int(np.floor(wd/2) + 1)
        aep, time = geo.compare_aep(num_terms=N, wd_resolution=360/wd, ws_avg=True, iter=5)
        flowers_terms[i] = N
        flowers_aep[i] = aep[0]
        flowers_time[i] = time[0]
        floris_bins[i] = wd
        floris_aep[i] = aep[1]
        floris_time[i] = time[1]

    flowers_aep /= flowers_aep[0]
    flowers_time /= flowers_time[0]
    floris_aep /= floris_aep[0]
    floris_time /= floris_time[0]

    ax1.plot(flowers_terms,flowers_time, '-o', markersize=3)
    ax2.plot(floris_bins,floris_time, '-o', markersize=3)
    ax3.plot(flowers_terms,flowers_aep, '-o', markersize=3)
    ax4.plot(floris_bins,floris_aep, '-o', markersize=3)

ax3.set(xlabel='Fourier Modes', ylabel='Normalized AEP')
ax1.set(ylabel='Normalized Time', title='FLOWERS')
ax4.set(xlabel='Wind Direction Bins')
ax2.set(title='Conventional')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

fig.tight_layout()
plt.show()
