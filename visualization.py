# FLOWERS

# Michael LoCascio

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import tools as tl


def plot_flow_field(fi, ax, bounds, pts=200, cmin=2, cmax=10):
    """
    Plots a filled contour map of the annually-averaged flow field.

    Args:
        fi: FLOWERS interface with valid Fourier coefficients
        ax: matplotlib axis handle to plot colormesh
        bounds (np.array): domain limits in the form 
            ([x_min, x_max], [y_min, y_max])
        pts: grid resolution (uniform in x and y)
        cmin: minimum wind speed for colorbar [m/s]
        cmax: maximum wind speed for colorbar [m/s]
    """

    # Enforce that fourier_coefficients() have been computed
    if not hasattr(fi, 'fs'):
        print(
            "Error, must compute Fourier coefficients before calculating wake"
        )
        return None
    
    # Define 2D grid based on defined domain limits
    xx = np.linspace(bounds[0,0], bounds[0,1], pts)
    yy = np.linspace(bounds[1,0], bounds[1,1], pts)
    XX, YY = np.meshgrid(xx, yy)

    # Freestream velocity component
    u0 = fi.fs.a_free[0] * np.pi

    # Superimpose wake velocity component from each turbine
    for i in range(len(fi.layout_x)):
        if i == 0:
            du = fi.calculate_wake(XX-fi.layout_x[i], YY-fi.layout_y[i])
        else:
            du += fi.calculate_wake(XX-fi.layout_x[i], YY-fi.layout_y[i])

    # Compute average wake velocity
    u = u0 - du

    # Mask points within rotor swept area
    zz = np.logical_or(np.isnan(u),np.sqrt((XX-fi.layout_x[0])**2 + (YY-fi.layout_y[0])**2) < fi.D/2)
    for j in range(len(fi.layout_x)-1):
        zz = np.logical_or(zz, np.sqrt((XX-fi.layout_x[j+1])**2 + (YY-fi.layout_y[j+1])**2) < fi.D/2)
    u_masked = np.ma.masked_where(zz,u)

    # Plot wake velocity colormesh and rotor swept areas
    im = ax.pcolormesh(XX, YY, u_masked, vmin=cmin, vmax=cmax, cmap='coolwarm')
    ax.plot(fi.layout_x, fi.layout_y, 'ow')

    return im

def plot_floris_field(fli, ax, wind_rose, bounds, pts=200, cmin=2, cmax=10):
    """
    Plots a filled contour map of the annually-averaged flow field using FLORIS.

    Args:
        fli: FLORIS interface
        fi: FLOWERS interface with valid Fourier coefficients
        ax: matplotlib axis handle to plot colormesh
        bounds (np.array): domain limits in the form 
            ([x_min, x_max], [y_min, y_max])
        pts: grid resolution (uniform in x and y)
        cmin: minimum wind speed for colorbar [m/s]
        cmax: maximum wind speed for colorbar [m/s]
    """

    # Resample wind rose by average wind speed to speed up plotting
    wr = wind_rose.copy(deep=True)
    wr = tl.resample_wind_direction(wr, wd=np.arange(0, 360, 5.0))
    wr = tl.resample_average_ws_by_wd(wr)

    # Numerically integrate over wind rose
    for i in range(len(wr.wd)): #range(len(wr.wd))

        # Redefine flow field for given wind speed and direction
        fli.reinitialize(wind_directions=[wr.wd[i]], wind_speeds=[wr.ws[i]])
        if i == 0:
            hor_plane = fli.calculate_horizontal_plane(
                height=90.0, 
                x_resolution = pts, 
                y_resolution = pts, 
                x_bounds=[bounds[0,0], bounds[0,1]], 
                y_bounds=[bounds[1,0], bounds[1,1]]
            )
            hor_plane.df.u = wr.freq_val[i] * hor_plane.df.u
        else:
            hor_plane1 = fli.calculate_horizontal_plane(
                height=90.0, 
                x_resolution = pts, 
                y_resolution = pts, 
                x_bounds=[bounds[0,0], bounds[0,1]], 
                y_bounds=[bounds[1,0], bounds[1,1]]
            )
            hor_plane.df.u += wr.freq_val[i] * hor_plane1.df.u
    
    # Reshape mesh grid for plotting
    x1_mesh = hor_plane.df.x1.values.reshape(hor_plane.resolution[1], hor_plane.resolution[0])
    x2_mesh = hor_plane.df.x2.values.reshape(hor_plane.resolution[1], hor_plane.resolution[0])
    u_mesh = hor_plane.df.u.values.reshape(hor_plane.resolution[1], hor_plane.resolution[0])

    # Mask rotor swept areas (TODO add rotor diameter variable)
    zz = np.logical_or(np.isnan(u_mesh),np.sqrt((x1_mesh-fli.floris.farm.layout_x[0])**2 + (x2_mesh-fli.floris.farm.layout_y[0])**2) < 126/2)
    for j in range(len(fli.floris.farm.layout_x)-1):
        zz = np.logical_or(zz, np.sqrt((x1_mesh-fli.floris.farm.layout_x[j+1])**2 + (x2_mesh-fli.floris.farm.layout_y[j+1])**2) < 126/2)
    u = np.ma.masked_where(zz, u_mesh)

    # Plot wake velocity colormesh and rotor swept areas
    im = ax.pcolormesh(x1_mesh,x2_mesh, u, cmap='coolwarm') #vmin=cmin, vmax=cmax,
    for j in range(len(fli.floris.farm.layout_x)):
        ax.scatter(fli.floris.farm.layout_x[j],fli.floris.farm.layout_y[j],c='white')

    return im

def plot_wind_rose(
        wind_rose,
        ax=None,
        color_map="viridis_r",
        ws_right_edges=np.array([5, 10, 15, 20, 25]),
        wd_bins=np.arange(0, 360, 15.0),
        legend_kwargs={},
    ):
        """
        Plots a wind rose showing the frequency of occurance
        of the specified wind direction and wind speed bins. If no axis is
        provided, a new one is created. (Copied from FLORIS)

        **Note**: Based on code provided by Patrick Murphy from the University
        of Colorado Boulder.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional): The figure axes
                on which the wind rose is plotted. Defaults to None.
            color_map (str, optional): Colormap to use. Defaults to 'viridis_r'.
            ws_right_edges (np.array, optional): The upper bounds of the wind
                speed bins (m/s). The first bin begins at 0. Defaults to
                np.array([5, 10, 15, 20, 25]).
            wd_bins (np.array, optional): The wind direction bin centers used
                for plotting (deg). Defaults to np.arange(0, 360, 15.).
            legend_kwargs (dict, optional): Keyword arguments to be passed to
                ax.legend().

        Returns:
            :py:class:`matplotlib.pyplot.axes`: A figure axes object containing
            the plotted wind rose.
        """
        # Resample data onto bins
        df_plot = tl.resample_wind_direction(wind_rose, wd=wd_bins)

        # Make labels for wind speed based on edges
        ws_step = ws_right_edges[1] - ws_right_edges[0]
        ws_labels = ["%d-%d m/s" % (w - ws_step, w) for w in ws_right_edges]

        # Grab the wd_step
        wd_step = wd_bins[1] - wd_bins[0]

        # Set up figure
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(polar=True))

        # Get a color array
        color_array = cm.get_cmap(color_map, len(ws_right_edges))

        for wd_idx, wd in enumerate(wd_bins):
            rects = list()
            df_plot_sub = df_plot[df_plot.wd == wd]
            for ws_idx, ws in enumerate(ws_right_edges[::-1]):
                plot_val = df_plot_sub[
                    df_plot_sub.ws <= ws
                ].freq_val.sum()  # Get the sum of frequency up to this wind speed
                rects.append(
                    ax.bar(
                        np.radians(wd),
                        plot_val,
                        width=0.9 * np.radians(wd_step),
                        color=color_array(ws_idx),
                        edgecolor="k",
                    )
                )
            # break

        # Configure the plot
        ax.legend(reversed(rects), ws_labels, **legend_kwargs)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_yticklabels([])

        return ax

def plot_optimal_layout(ax, boundaries, x_final, y_final, x_init, y_init, D):

    # Plot turbine locations
    ax.plot(x_init / D, y_init / D, "ob")
    ax.plot(x_final / D, y_final / D, "or")
    ax.set(xlabel="x / D", ylabel="y / D", aspect='equal')
    ax.grid()

    # Plot plant boundary
    verts = boundaries / D
    for i in range(len(verts)):
        if i == len(verts) - 1:
            ax.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "black")
        else:
            ax.plot(
                [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "black"
            )