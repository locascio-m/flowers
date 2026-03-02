# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd
import tools as tl

class FlowersInterface():
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
        num_terms (int, optional): number of Fourier modes
        k (float, optional): wake expansion rate
        turbine (str, optional): turbine type:
                - 'nrel_5MW' (default)

    """

    ###########################################################################
    # Initialization tools
    ###########################################################################

    def __init__(self, wind_rose, layout_x, layout_y, num_terms=0, k=0.05, turbine=None):

        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.k = k

        if turbine is None or turbine == 'nrel_5MW':
            self.turbine = 'nrel_5MW'
            self.D = 126.
            self.U = 25.0
        
        self._fourier_coefficients(num_terms=num_terms)
    
    def reinitialize(self, wind_rose=None, layout_x=None, layout_y=None, num_terms=None, k=None):

        if wind_rose is not None:
            self.wind_rose = wind_rose
            self._fourier_coefficients(num_terms=num_terms)
        
        if num_terms is not None:
            self._fourier_coefficients(num_terms=num_terms)
        
        if layout_x is not None:
            self.layout_x = layout_x
        
        if layout_y is not None:
            self.layout_y = layout_y
        
        if k is not None:
            self.k = k
    
    ###########################################################################
    # User functions
    ###########################################################################

    def get_layout(self):
        return self.layout_x, self.layout_y
    
    def get_wind_rose(self):
        return self.wind_rose
    
    def get_num_modes(self):
        return len(self.fs)
    
    def calculate_aep(self, gradient=False):
        """
        Compute farm AEP (and Cartesian gradients) for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
            gradient (numpy.array(float)): (dAEP/dx, dAEP/dy) for each turbine [Wh/m]
        """
        
        # Power component from freestream
        u0 = self.fs.c[0]

        # Normalize and reshape relative positions into symmetric 2D array
        xx = (self.layout_x - np.reshape(self.layout_x,(-1,1)))/self.D
        yy = (self.layout_y - np.reshape(self.layout_y,(-1,1)))/self.D

        # Convert to normalized polar coordinates
        R = np.sqrt(xx**2 + yy**2)
        THETA = np.arctan2(yy,xx) / (2 * np.pi)

        # Set up mask for rotor swept area
        mask_area = np.array(R <= 0.5, dtype=int)
        mask_val = self.fs.c[0]

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / (2*R) + self.k * np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            / (-self.k / (2*R) + np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            ) / (2 * np.pi)
        theta_c = np.nan_to_num(theta_c)
        
        # Contribution from zero-frequency Fourier mode
        du = self.fs.a[0] * theta_c / (2 * self.k * R + 1)**2 * (
            1 + (8 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (2 * self.k * R + 1)))
        
        # Initialize gradient and calculate zero-frequency modes
        if gradient == True:
            grad = np.zeros((len(self.layout_x),2))

            # Change in theta_c wrt radius
            dtdr = (-1 / (4 * np.pi * R**2 * np.sqrt(self.k**2 - (2*R)**(-2) + 1)))
            dtdr = np.nan_to_num(dtdr)

            # Zero-frequency mode of change in power deficit wrt radius
            dpdr = (-4 * self.fs.a[0] * self.k * theta_c * (3 + 6 * self.k * R + 2 * np.pi**2 * (4 * self.k * R - 1) * theta_c**2) + 
                    3 * self.fs.a[0] * (1 + 2 * self.k * R) * (1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr) / (
                3 * (1 + 2*self.k*R)**4)

        # Reshape variables for vectorized calculations
        # m = np.arange(1, len(self.fs.b))
        # a = np.swapaxes(np.tile(np.expand_dims(self.fs.a[1:], axis=(1,2)),np.shape(R.T)),0,2)
        # b = np.swapaxes(np.tile(np.expand_dims(self.fs.b[1:], axis=(1,2)),np.shape(R.T)),0,2)
        # R = np.tile(np.expand_dims(R, axis=2),len(m))
        # THETA = np.tile(np.expand_dims(THETA, axis=2),len(m))
        # theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(m))

        # # Vectorized contribution of higher Fourier modes
        # du += np.sum((1 / (np.pi * m * (2 * self.k * R + 1)**2) * (
        #     a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) * (
        #         np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
        #             ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)
    
        for m in np.arange(1,len(self.fs.b)):
            du += (1 / (np.pi * m * (2 * self.k * R + 1)**2) * (
            self.fs.a[m] * np.cos(2 * np.pi * m * THETA) + self.fs.b[m] * np.sin(2 * np.pi * m * THETA)) * (
                np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                    ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c))))

        if gradient==True:
            dpdt = 0
            for m in np.arange(1,len(self.fs.b)):
                # Higher Fourier modes of change in power deficit wrt angle
                dpdt += (2 / (2 * self.k * R + 1)**2 * (
                    self.fs.b[m] * np.cos(2 * np.pi * m * THETA) - self.fs.a[m] * np.sin(2 * np.pi * m * THETA)) * (
                        np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                            ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c))))

                # Higher Fourier modes of change in power deficit wrt radius
                dpdr += ((self.fs.a[m] * np.cos(2 * np.pi * m * THETA) + self.fs.b[m] * np.sin(2 * np.pi * m * THETA)) / (np.pi * m**3 * (2 * self.k * R + 1)**4) * (
                    -4 * self.k * np.sin(2 * np.pi * m * theta_c) * (1 + m**2 + 2 * self.k * R * (m**2 - 2) + 2 * np.pi**2 * m**2 * (4 * self.k * R - 1) * theta_c**2) + 
                    2 * np.pi * m * np.cos(2 * np.pi * m * theta_c) * (4 * self.k * (1 - 4 * self.k * R) * theta_c + m**2 * (2 * self.k * R + 1) * (
                    1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr)))

            # dtdr = np.tile(np.expand_dims(dtdr, axis=2),len(m))
            
            # # Higher Fourier modes of change in power deficit wrt angle
            # dpdt = np.sum((2 / (2 * self.k * R + 1)**2 * (
            #     b * np.cos(2 * np.pi * m * THETA) - a * np.sin(2 * np.pi * m * THETA)) * (
            #         np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
            #             ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)

            # # Higher Fourier modes of change in power deficit wrt radius
            # dpdr += np.sum(((a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) / (np.pi * m**3 * (2 * self.k * R + 1)**4) * (
            #     -4 * self.k * np.sin(2 * np.pi * m * theta_c) * (1 + m**2 + 2 * self.k * R * (m**2 - 2) + 2 * np.pi**2 * m**2 * (4 * self.k * R - 1) * theta_c**2) + 
            #     2 * np.pi * m * np.cos(2 * np.pi * m * theta_c) * (4 * self.k * (1 - 4 * self.k * R) * theta_c + m**2 * (2 * self.k * R + 1) * (
            #     1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr))), axis=2)
            
        # Apply mask for points within rotor radius
        du = du * (1 - mask_area) + mask_val * mask_area
        np.fill_diagonal(du, 0.)
        
        # Sum power for each turbine
        du = np.sum(du, axis=1)
        aep = np.sum((u0 - du)**3)
        aep *= np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760

        # Complete gradient calculation
        if gradient==True:
            dx = xx/R*dpdr + -yy/(2*np.pi*R**2)*dpdt
            dy = yy/R*dpdr + xx/(2*np.pi*R**2)*dpdt
            # dx = xx/np.sqrt(xx**2+yy**2)*dpdr + -yy/(2*np.pi*(xx**2+yy**2))*dpdt
            # dy = yy/np.sqrt(xx**2+yy**2)*dpdr + xx/(2*np.pi*(xx**2+yy**2))*dpdt

            dx = np.nan_to_num(dx)
            dy = np.nan_to_num(dy)
            
            coeff = (u0 - du)**2
            for i in range(len(grad)):
                # Isolate gradient to turbine 'i'
                grad_mask = np.zeros_like(xx)
                grad_mask[i,:] = -1.
                grad_mask[:,i] = 1.

                grad[i,0] = np.sum(coeff*np.sum(dx*grad_mask,axis=1)) 
                grad[i,1] = np.sum(coeff*np.sum(dy*grad_mask,axis=1))

            grad *= -3 * np.pi / 8 * 1.225 * self.D * self.U**3 * 8760

            return aep, grad
        
        else:
            return aep

    ###########################################################################
    # Private functions
    ###########################################################################

    def _fourier_coefficients(self, num_terms=0):
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
        # wr.loc[len(wr)] = wr.iloc[0]
        # wr.freq_val /= np.sum(wr.freq_val)

        # Normalize wind speed by cut-out speed
        wr["ws"] /= self.U

        # Look up thrust and power coefficients for each wind direction bin
        ct = tl.ct_lookup(wr.ws,self.turbine)
        cp = tl.cp_lookup(wr.ws,self.turbine)

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