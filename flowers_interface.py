# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools as tl


class Flowers():
    """
    Flowers is a high-level user interface to the FLOWERS AEP model.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        k (float, optional): wake expansion rate
        D (float, optional): rotor diameter [m]
        U (float, optional): cut-out speed [m/s]

    """

    def __init__(self, wind_rose, layout_x, layout_y, k=0.05, D=126.0, U=25.0):

        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.k = k
        self.D = D
        self.U = U

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
                [1, floor(num_wind_directions/2)]
        
        """

        # Resample wind rose for average wind speed per wind direction
        wr = self.wind_rose.copy(deep=True)
        wr = tl.resample_average_ws_by_wd(wr)

        # Transform wind direction to polar angle 
        wr["wd"] = np.remainder(450 - wr.wd, 360)
        wr.sort_values("wd", inplace=True)
        # wr = wr.append(wr.iloc[0])
        wr.loc[len(wr)] = wr.iloc[0]
        wr.freq_val /= np.sum(wr.freq_val)

        # Normalize wind speed by cut-out speed
        wr["ws"] /= self.U

        # Look up thrust and power coefficients for each wind direction bin
        ct = tl.ct_lookup(wr.ws)
        cp = tl.cp_lookup(wr.ws)

        # Average freestream term
        c = np.sum(cp**(1/3) * wr.ws * wr.freq_val)

        # Fourier expansion of wake deficit term
        c1 = cp**(1/3) * (1 - np.sqrt(1 - ct)) * wr.ws * wr.freq_val
        c1ft = 2 * np.fft.rfft(c1)
        a =  c1ft.real
        b = -c1ft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a):
            a = a[0:num_terms]
            b = b[0:num_terms]

        # Compile Fourier coefficients
        self.fs = pd.DataFrame({'a': a, 'b': b, 'c': c})

    def calculate_aep(self):
        """
        Compute farm AEP for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
        """

        # Power component from freestream
        u0 = self.fs.c[0]

        # Reshape relative positions into symmetric 2D array
        xx = self.layout_x - np.reshape(self.layout_x,(-1,1))
        yy = self.layout_y - np.reshape(self.layout_y,(-1,1))

        # Vectorized wake calculation
        du = self._calculate_wake(xx,yy)

        # Mask turbine interaction with itself
        np.fill_diagonal(du, 0.)

        # Sum power for each turbine
        du = np.sum(du, axis=1)
        aep = np.sum((u0 - du)**3)
        aep *= np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760
        
        return aep

    def calculate_field(self, X, Y):
        """
        Compute flow field speed for a given 2D mesh.
        
        Args:
            X (np.array(float)): 2D array of the x-position of each point in
                the flow field of interest.
            Y (np.array(float)): 2D array of the y-position of each point in
                the flow field of interest.

        Returns:
            u (np.array(float)): annually-averaged flow speed at each point
                in the 2D domain.

        """

        # Calculate average freestream wind speed
        u0 = self.fs.c[0]

        # Reshape relative position between each point and each turbine
        x = X.flatten()
        y = Y.flatten()
        matrix_x = self.layout_x - np.reshape(x, (-1,1))
        matrix_y = self.layout_y - np.reshape(y, (-1,1))
        matrix_r = np.sqrt(matrix_x**2 + matrix_y**2)

        # Compute wake velocity deficits
        du = self._calculate_wake(matrix_x,matrix_y)

        # Mask points within rotor swept area
        du = np.ma.masked_where(matrix_r < self.D/2, du)

        # Linear superposition of velocity deficits
        du_sum = np.sum(du, axis=1)
        u = self.U * (u0 - du_sum)
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

        # Convert to normalized polar coordinates
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y,X) / (2 * np.pi)

        # Set up mask for rotor swept area
        mask_area = np.array(R <= 1, dtype=int)
        mask_val = self.fs.c[0]

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / R + self.k * np.sqrt(1 + self.k**2 - R**(-2)))
            / (-self.k / R + np.sqrt(1 + self.k**2 - R**(-2)))
            ) / (2 * np.pi)
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.fs.a[0] * theta_c / (self.k * R + 1)**2 * (
            1 + (4 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (self.k * R + 1)))

        # Reshape variables for vectorized calculations
        n = np.arange(1, len(self.fs.b))
        a = np.swapaxes(np.tile(np.expand_dims(self.fs.a[1:], axis=(1,2)),np.shape(R.T)),0,2)
        b = np.swapaxes(np.tile(np.expand_dims(self.fs.b[1:], axis=(1,2)),np.shape(R.T)),0,2)
        R = np.tile(np.expand_dims(R, axis=2),len(n))
        THETA = np.tile(np.expand_dims(THETA, axis=2),len(n))
        theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(n))

        # Vectorized contribution of higher Fourier modes
        du += np.sum((1 / (np.pi * n * (self.k * R + 1)**2) * (
            a * np.cos(2 * np.pi * n * THETA) + b * np.sin(2 * np.pi * n * THETA)) * (
                np.sin(2 * np.pi * n * theta_c) + self.k * R / (n**2 * (self.k * R + 1)) * (
                    ((2 * np.pi * theta_c * n)**2 - 2) * np.sin(2 * np.pi * n * theta_c) + 4*np.pi*n*theta_c*np.cos(2 * np.pi * n * theta_c)))), axis=2)
        
        # Apply mask for points within rotor radius
        du = du * (1 - mask_area) + mask_val * mask_area

        return du


    def fourier_coefficients_modified(self, num_terms=0):
        """
        Compute the Fourier series expansion coefficients from the wind rose.
        Modifies the Flowers interface in place to add a Fourier coefficients
        dataframe:
            fs (pandas:dataframe): Fourier coefficients used to expand the wind rose:
                - 'a0': real coefficients of zero-order (i.e. freestream) power term
                - 'a1': real coefficients of first-order power term
                - 'b1': imaginary coefficients of first-order power term
                - 'a2': real coefficients of second-order power term
                - 'b2': imaginary coefficients of second-order power term

        Args:
            num_terms (int, optional): the number of Fourier modes to save in the range
                [1, floor(num_wind_directions/2)]
        
        """

        # Resample wind rose for average wind speed per wind direction
        wr = self.wind_rose.copy(deep=True)
        # cp_tmp = tl.cp_lookup(wr.ws)
        # a0_tmp = 2 * np.sum(cp_tmp * wr.freq_val * wr.ws**3)
        wr = tl.resample_average_ws_by_wd(wr)

        # Transform wind direction to polar angle 
        wr["wd"] = np.remainder(450 - wr.wd, 360)
        wr.sort_values("wd", inplace=True)
        wr = wr.append(wr.iloc[0])
        wr.freq_val /= np.sum(wr.freq_val)

        # Normalize wind speed by cut-out speed
        wr["ws"] /= self.U

        # Look up thrust and power coefficients for each wind direction bin TODO: remove (1/3) here
        ct = tl.ct_lookup(wr.ws, ct=0.75)
        # cp = tl.cp_lookup(wr.ws)
        
        # Define functions for Fourier expansions TODO: add cube here
        c0 = wr.ws * wr.freq_val
        c1 = wr.ws * (1 - np.sqrt(1 - ct)) * wr.freq_val
        # c0 = cp * wr.freq_val * wr.ws**3
        # c1 = cp * (1 - np.sqrt(1 - ct)) * wr.ws**3 * wr.freq_val
        # c2 = cp * wr.freq_val * wr.ws**3 * (1 - np.sqrt(1 - ct))**2
        # c3 = cp * wr.freq_val * wr.ws**3 * (1 - np.sqrt(1 - ct))**3

        # Fourier expansions TODO: are these coefficients calculated correctly? (norm='forward')
        c0ft = 2 * np.fft.rfft(c0)
        a0 = c0ft.real[0]
        # print(a0)

        c1ft = 2 * np.fft.rfft(c1)
        a1 = c1ft.real
        b1 = -c1ft.imag

        # c2ft = 2 * np.fft.rfft(c2)
        # a2 = c2ft.real
        # b2 = -c2ft.imag

        # c3ft = 2 * np.fft.rfft(c3)
        # a3 = c3ft.real
        # b3 = -c3ft.imag

        # Truncate Fourier series
        if num_terms > 0 and num_terms <= len(a1):
            a1 = a1[0:num_terms]
            b1 = b1[0:num_terms]
            # a2 = a2[0:num_terms]
            # b2 = b2[0:num_terms]
            # a3 = a3[0:num_terms]
            # b3 = b3[0:num_terms]
        
        self.fs = pd.DataFrame({
            'a0': a0, #a0_tmp
            'a1': a1,
            'b1': b1,
            # 'a2': a2,
            # 'b2': b2,
            # 'a3': a3,
            # 'b3': b3,
            })

    def calculate_aep_modified(self):
        """
        Compute farm AEP for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]

        """

        # Power component from freestream
        p0 = self.fs.a0[0] / 2

        # Reshape relative positions into symmetric 2D array
        X = self.layout_x - np.reshape(self.layout_x,(-1,1))
        Y = self.layout_y - np.reshape(self.layout_y,(-1,1)) 

        # Normalize positions by rotor radius
        X /= (self.D/2)
        Y /= (self.D/2)

        # Convert to normalized polar coordinates
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y,X)/(2 * np.pi)

        # Handle two edge cases: standalone turbine and colliding turbines TODO: remove cube here
        R = np.ma.masked_where(np.eye(len(R)) == 1,R)
        if len(R) == 1:
            p = np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760 * tl.cp_lookup(p0, cp=0.43) * p0**3
        elif np.any(R < 1.0):
            p = 0.
        else:
            # Critical polar angle of wake edge 1 / (2 * np.pi) * 
            # theta_c = np.arctan(
            # (1 / R + self.k * np.sqrt(1 + self.k**2 - R**(-2)))
            # / (-self.k / R + np.sqrt(1 + self.k**2 - R**(-2)))
            # )
            # theta_c = np.nan_to_num(theta_c)
            theta_c = np.arctan(
                (1 / R + self.k * np.sqrt(1 + self.k**2 - R**(-2)))
                / (-self.k / R + np.sqrt(1 + self.k**2 - R**(-2)))
                )/(2 * np.pi)
            theta_c = np.nan_to_num(theta_c)
            
            # Contribution from zero-frequency Fourier mode
            # p1 = self.fs.a1[0] * (
            # theta_c * (self.k * R * (theta_c**2 + 3) + 3) / (3 * (self.k * R + 1)**3)
            # )
            p1 = self.fs.a1[0] * theta_c / (self.k * R + 1)**2 * (
                1 + (4 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (self.k * R + 1)))
            # p2 = self.fs.a2[0] * theta_c / (self.k * R + 1)**4 * (
            #     1 + (8 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (self.k * R + 1)))
            # p3 = self.fs.a3[0] * theta_c / (self.k * R + 1)**6 * (
            #     1 + (12 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (self.k * R + 1)))

            # Reshape variables for vectorized calculations ((turbine, turbine wakes, Fourier modes))
            n = np.arange(1, len(self.fs.b1))
            a1 = np.swapaxes(np.tile(np.expand_dims(self.fs.a1[1:], axis=(1,2)),np.shape(R)),0,2)
            b1 = np.swapaxes(np.tile(np.expand_dims(self.fs.b1[1:], axis=(1,2)),np.shape(R)),0,2)
            # a2 = np.swapaxes(np.tile(np.expand_dims(self.fs.a2[1:], axis=(1,2)),np.shape(R)),0,2)
            # b2 = np.swapaxes(np.tile(np.expand_dims(self.fs.b2[1:], axis=(1,2)),np.shape(R)),0,2)
            # a3 = np.swapaxes(np.tile(np.expand_dims(self.fs.a3[1:], axis=(1,2)),np.shape(R)),0,2)
            # b3 = np.swapaxes(np.tile(np.expand_dims(self.fs.b3[1:], axis=(1,2)),np.shape(R)),0,2)
            R = np.tile(np.expand_dims(R, axis=2),len(n))
            THETA = np.tile(np.expand_dims(THETA, axis=2),len(n))
            theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(n))
            

            # Vectorized contribution of higher Fourier modes
            # p1 += np.sum((2 * (
            # a1 * np.cos(n * THETA) + b1 * np.sin(n * THETA))
            # ) / (n * (self.k * R + 1))**3 * (
            #     np.sin(n * theta_c) 
            #     * (n**2 * (self.k * R * (theta_c**2 + 1) + 1) - 2 * self.k * R) 
            #     + 2 * n * self.k * R * theta_c * np.cos(n * theta_c)
            #     ), axis=2)
            p1 += np.sum((1 / (np.pi * n * (self.k * R + 1)**2) * (
                a1 * np.cos(2 * np.pi * n * THETA) + b1 * np.sin(2 * np.pi * n * THETA)) * (
                    np.sin(2 * np.pi * n * theta_c) + self.k * R / (n**2 * (self.k * R + 1)) * (
                        ((2 * np.pi * theta_c * n)**2 - 2) * np.sin(2 * np.pi * n * theta_c) + 4*np.pi*n*theta_c*np.cos(2 * np.pi * n * theta_c)))), axis=2)
            # p2 += np.sum((1 / (np.pi * n * (self.k * R + 1)**4) * (
            #     a2 * np.cos(2 * np.pi * n * THETA) + b2 * np.sin(2 * np.pi * n * THETA)) * (
            #         np.sin(2 * np.pi * n * theta_c) + 2 * self.k * R / (n**2 * (self.k * R + 1)) * (
            #             ((2 * np.pi * theta_c * n)**2 - 2) * np.sin(2 * np.pi * n * theta_c) + 4*np.pi*n*theta_c*np.cos(2 * np.pi * n * theta_c)))), axis=2)
            # p3 += np.sum((1 / (np.pi * n * (self.k * R + 1)**6) * (
            #     a3 * np.cos(2 * np.pi * n * THETA) + b3 * np.sin(2 * np.pi * n * THETA)) * (
            #         np.sin(2 * np.pi * n * theta_c) + 3 * self.k * R / (n**2 * (self.k * R + 1)) * (
            #             ((2 * np.pi * theta_c * n)**2 - 2) * np.sin(2 * np.pi * n * theta_c) + 4*np.pi*n*theta_c*np.cos(2 * np.pi * n * theta_c)))), axis=2)
            # p00 = -3 * p1 + 3 * p2 #- p3 # TODO: replace combination here
            # p = np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760 * (p0 + np.sum(p00, axis=1))
            p00 = -p1
            p = np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760 * tl.cp_lookup(p0, cp=0.43) * (p0 + np.sum(p00, axis=1))**3 #TODO: remove cube
            # print(p)
            # TODO: when adding ws**3 to FFT and removing cube here the code breaks

        # Sum power for each turbine 
        aep = np.sum(p)

        return aep