import numpy as np
import matplotlib.pyplot as plt
import tools as tl

import flowers_interface as flow
import visualization as vis

# # Realistic
# wr = tl.load_wind_rose(1)
# wr = tl.resample_average_ws_by_wd(wr)
# wd = wr.wd
# freq = wr.freq_val

# wd = np.append(wd, 0.)
# freq = np.append(freq, freq[0])
# freq /= np.sum(freq)

# fig = plt.figure(figsize=(11,5))
# ax0 = fig.add_subplot(121, polar=True)
# ax1 = fig.add_subplot(122, polar=True)

# # Configure the plot
# ax0.set_theta_direction(-1)
# ax0.set_theta_offset(np.pi / 2.0)
# ax0.set_theta_zero_location("N")
# ax0.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax0.set_yticklabels([])

# ax1.set_theta_direction(-1)
# ax1.set_theta_offset(np.pi / 2.0)
# ax1.set_theta_zero_location("N")
# ax1.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax1.set_yticklabels([])

# ax0.plot(wd * np.pi / 180, freq, 'k')

# f0 = np.fft.irfft(np.fft.rfft(freq)[:],361)
# f1 = np.fft.irfft(np.fft.rfft(freq)[:100],361)
# f2 = np.fft.irfft(np.fft.rfft(freq)[:50],361)
# f3 = np.fft.irfft(np.fft.rfft(freq)[:25],361)
# f4 = np.fft.irfft(np.fft.rfft(freq)[:10],361)
# ax1.plot(wd * np.pi / 180, freq, 'k')
# ax1.plot(wd * np.pi / 180, f0, '--')
# ax1.plot(wd * np.pi / 180, f1, '--')
# ax1.plot(wd * np.pi / 180, f2, '--')
# ax1.plot(wd * np.pi / 180, f3, '--')
# ax1.plot(wd * np.pi / 180, f4, '--')
# ax1.legend(['Real','N = 182','N = 100','N = 50','N = 25','N = 10'])


# wr = tl.load_wind_rose(2)
# wr = tl.resample_average_ws_by_wd(wr)
# wd = wr.wd
# freq = wr.freq_val

# wd = np.append(wd, 0.)
# freq = np.append(freq, freq[0])
# freq /= np.sum(freq)

# fig = plt.figure(figsize=(11,5))
# ax0 = fig.add_subplot(121, polar=True)
# ax1 = fig.add_subplot(122, polar=True)

# # Configure the plot
# ax0.set_theta_direction(-1)
# ax0.set_theta_offset(np.pi / 2.0)
# ax0.set_theta_zero_location("N")
# ax0.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax0.set_yticklabels([])

# ax1.set_theta_direction(-1)
# ax1.set_theta_offset(np.pi / 2.0)
# ax1.set_theta_zero_location("N")
# ax1.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax1.set_yticklabels([])

# ax0.plot(wd * np.pi / 180, freq, 'k')

# f0 = np.fft.irfft(np.fft.rfft(freq)[:],361)
# f1 = np.fft.irfft(np.fft.rfft(freq)[:100],361)
# f2 = np.fft.irfft(np.fft.rfft(freq)[:50],361)
# f3 = np.fft.irfft(np.fft.rfft(freq)[:25],361)
# f4 = np.fft.irfft(np.fft.rfft(freq)[:10],361)
# ax1.plot(wd * np.pi / 180, freq, 'k')
# ax1.plot(wd * np.pi / 180, f0, '--')
# ax1.plot(wd * np.pi / 180, f1, '--')
# ax1.plot(wd * np.pi / 180, f2, '--')
# ax1.plot(wd * np.pi / 180, f3, '--')
# ax1.plot(wd * np.pi / 180, f4, '--')
# ax1.legend(['Real','N = 182','N = 100','N = 50','N = 25','N = 10'])

# # Artificial
# wd = np.arange(0., 360., 1.)
# freq = (np.exp(-(wd-270)**2/(2*10)**2))

# wd = np.append(wd, 0.)
# freq = np.append(freq, freq[0])
# freq /= np.sum(freq)

# fig = plt.figure(figsize=(11,5))
# ax0 = fig.add_subplot(121, polar=True)
# ax1 = fig.add_subplot(122, polar=True)

# # Configure the plot
# ax0.set_theta_direction(-1)
# ax0.set_theta_offset(np.pi / 2.0)
# ax0.set_theta_zero_location("N")
# ax0.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax0.set_yticklabels([])

# ax1.set_theta_direction(-1)
# ax1.set_theta_offset(np.pi / 2.0)
# ax1.set_theta_zero_location("N")
# ax1.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# # ax1.set_yticklabels([])

# ax0.plot(wd * np.pi / 180, freq, 'k')

# f0 = np.fft.irfft(np.fft.rfft(freq)[:],361)
# f1 = np.fft.irfft(np.fft.rfft(freq)[:100],361)
# f2 = np.fft.irfft(np.fft.rfft(freq)[:50],361)
# f3 = np.fft.irfft(np.fft.rfft(freq)[:25],361)
# f4 = np.fft.irfft(np.fft.rfft(freq)[:10],361)
# ax1.plot(wd * np.pi / 180, freq, 'k')
# ax1.plot(wd * np.pi / 180, f0, '--')
# ax1.plot(wd * np.pi / 180, f1, '--')
# ax1.plot(wd * np.pi / 180, f2, '--')
# ax1.plot(wd * np.pi / 180, f3, '--')
# ax1.plot(wd * np.pi / 180, f4, '--')
# ax1.legend(['Real','N = 182','N = 100','N = 50','N = 25','N = 10'])


##
x0 = np.linspace(0., 4*7*126., 5)
y0 = np.linspace(0., 2*7*126., 3)
layout_x, layout_y = np.meshgrid(x0,y0)
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()
wr = tl.load_wind_rose(6)
x = np.linspace(-7.*126.,35.*126.,200)
y = np.linspace(-7.*126.,21.*126.,100)
XX,YY = np.meshgrid(x,y)

fi = flow.Flowers(wr,layout_x,layout_y,k=0.05,D=126.)
fi.fourier_coefficients_legacy()
U = fi.calculate_field(XX,YY)

fig = plt.figure(figsize=(13,5))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, polar=True)
im = ax0.contourf(XX,YY,U,20,cmap='coolwarm')
for xx,yy in zip(layout_x,layout_y):
    ax0.add_patch(plt.Circle((xx, yy), 126./2, color='white'))
cbar = fig.colorbar(im,ax=ax0,location='top')
ax0.set(xlabel='x [m]', ylabel='y [m]', aspect='equal')
cbar.ax.set(xlabel='$\overline{u}$ [m/s]')
vis.plot_wind_rose(wr,ax1)
fig.tight_layout()

plt.savefig('wesc_abstract', dpi=500)

plt.show()