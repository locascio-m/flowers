# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd

import tools as tl


class Flowers():
    """
    Flowers is a high-level user interface to the FLOWERS model.
    It can compute either the annually-averaged flow field or 
    annual energy production.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        k (float): wake expansion rate
        D (float): rotor diameter [m]

    """

    def __init__(self, wind_rose, layout_x, layout_y, k=0.05, D=126.0):

        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.k = k
        self.D = D

    def fourier_coefficients(self, num_terms=0):
        """
        Compute the Fourier series expansion coefficients from the wind rose.
        Modifies the Flowers interface in place to add a Fourier coefficients
        dataframe:
            fs (pandas:dataframe): Fourier coefficients used to expand the wind rose:
                - 'a_free': real coefficients of freestream component
                - 'a_wake': real coefficients of wake component
                - 'b_wake': imaginary coefficients of wake component

        Args:
            num_terms (int, optional): the number of Fourier modes to save in the range
                [1, ceiling(num_wind_directions/2)]
        
        """

        # Resample wind rose for average wind speed per wind direction
        wr = self.wind_rose.copy(deep=True)
        wr = tl.resample_average_ws_by_wd(wr)

        # Transform wind direction to polar angle 
        wr["wd"] = np.remainder(450 - wr.wd, 360)
        wr.sort_values("wd", inplace=True)
        wr = wr.append(wr.iloc[0])

        # Look up thrust coefficient for each wind direction bin
        ct = np.zeros(len(wr.ws))
        ct = tl.ct_lookup(wr.ws)
        ct = ct * np.array(ct < 1) + np.ones_like(ct) * np.array(ct >= 1) * 0.99999
        # ct = np.zeros(len(wr.ws))
        # for wd in range(len(wr.ws)):
        #     ct[wd] = tl.ct_lookup(wr.ws[wd])
        #     if ct[wd] >= 1.0:
        #         ct[wd] = 0.99999
        
        # Fourier expansion of freestream term 
        g = 1 / (2 * np.pi) * wr.ws * wr.freq_val
        gft = 2 * np.fft.rfft(g)
        a_free =  gft.real

        # Fourier expansion of wake deficit term
        h = 1 / (2 * np.pi) * (1 - np.sqrt(1 - ct)) * wr.ws * wr.freq_val
        hft = 2 * np.fft.rfft(h)
        a_wake =  hft.real
        b_wake = -hft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a_free):
            a_free = a_free[0:num_terms]
            a_wake = a_wake[0:num_terms]
            b_wake = b_wake[0:num_terms]

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a_free': a_free, 'a_wake': a_wake, 'b_wake': b_wake})

    def calculate_aep(self):
        """
        Compute farm AEP for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]

        """

        # Power component from freestream
        p0 = self.fs.a_free[0] * np.pi

        # Reshape relative positions into symmetric 2D array
        matrix_x = self.layout_x - np.reshape(self.layout_x,(-1,1))
        matrix_y = self.layout_y - np.reshape(self.layout_y,(-1,1))
        # matrix_r = np.sqrt(matrix_x**2 + matrix_y**2)

        # Vectorized wake calculation
        p = self._calculate_wake(matrix_x,matrix_y)

        # Mask turbine interaction with itself
        np.fill_diagonal(p, 0.)

        # Sum power for each turbine 
        p_sum = np.sum(p, axis=1)
        aep = np.sum(tl.cp_lookup(p0 - p_sum)  * (p0 - p_sum)**3)
        aep *= 0.5 * 1.225 * np.pi * self.D**2 / 4 * 8760
        
        return aep

    def calculate_field(self, X, Y):
        """
        Compute flow field speed for a given 2D mesh.
        
        Args:
            aep (float): farm AEP [Wh]

        Returns:
            aep (float): farm AEP [Wh]

        """
        u0 = self.fs.a_free[0] * np.pi
        x = X.flatten()
        y = Y.flatten()
        matrix_x = np.reshape(x, (-1,1)) - self.layout_x
        matrix_y = np.reshape(y, (-1,1)) - self.layout_y
        matrix_r = np.sqrt(matrix_x**2 + matrix_y**2)
        # TODO: fix masking
        du = self._calculate_wake(matrix_x,matrix_y)
        du = np.ma.masked_where(matrix_r < self.D/2, du)
        du_sum = np.sum(du, axis=1)
        u = u0 - du_sum
        
        u = np.reshape(u, np.shape(X))

        return u

    def _calculate_wake(self, X, Y):
        """
        Private method to compute the annually-averaged velocity deficit
            at a given set of points.

        Args:
            X (np.array(float)): 2D array in the shape (num_turbines, num_field_pts)
                containing the x-position of each point relative to the turbine
                in each row.
            Y (np.array(float)): 2D array in the shape (num_turbines, num_field_pts)
                containing the y-position of each point relative to the turbine
                in each row.
        
        Returns:
            du (np.array(float)): 2D array in the shape (num_turbines, num_field_pts)
                containing the average wake velocity deficit at each point.

        """

        # Enforce that the Fourier coefficients have been computed
        if not hasattr(self, 'fs'):
            raise RuntimeError(
                "Must compute Fourier coefficients before calculating wake"
                )

        # Normalize positions by rotor radius
        X /= self.D/2
        Y /= self.D/2

        # Convert to polar coordinates
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y,X) + np.pi

        # Set up mask for rotor swept area
        mask_area = np.array(R <= 1, dtype=int)
        mask_val = self.fs.a_free[0] * np.pi

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.abs(np.arctan(
            (1 / R + self.k * np.sqrt(1 + self.k**2 - R**(-2)))
            / (-self.k / R + np.sqrt(1 + self.k**2 - R**(-2)))
            ))
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.fs.a_wake[0] * (
            theta_c * (self.k * R * (theta_c**2 + 3) + 3) / (3 * (self.k * R + 1)**3)
            )

        # Reshape variables for vectorized calculations (num_turbines, num_field_pts)
        n = np.arange(1, len(self.fs.b_wake))
        a_wake = np.swapaxes(np.tile(np.expand_dims(self.fs.a_wake[1:], axis=(1,2)),np.shape(R.T)),0,2)
        b_wake = np.swapaxes(np.tile(np.expand_dims(self.fs.b_wake[1:], axis=(1,2)),np.shape(R.T)),0,2)
        R = np.tile(np.expand_dims(R, axis=2),len(n))
        THETA = np.tile(np.expand_dims(THETA, axis=2),len(n))
        theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(n))

        # Vectorized contribution of higher Fourier modes
        du += np.sum((2 * (
            a_wake * np.cos(n * THETA) + b_wake * np.sin(n * THETA))
            ) / (n * (self.k * R + 1))**3 * (
                np.sin(n * theta_c) 
                * (n**2 * (self.k * R * (theta_c**2 + 1) + 1) - 2 * self.k * R) 
                + 2 * n * self.k * R * theta_c * np.cos(n * theta_c)
                ), axis=2)
        
        du = du * (1 - mask_area) + mask_val * mask_area

        return du