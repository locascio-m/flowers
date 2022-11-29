# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

import floris.tools.wind_rose as rose


###########################################################################
# Layout generation
###########################################################################

def random_layout(boundaries=[], n_turb=0, D=126.0, min_dist=2.0, idx=None):
    """
    Generate a random wind farm layout within the specified boundaries.
    Minimum spacing between turbines is 2D.

    Args:
        boundaries (list(tuple)): boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)]
        n_turb (int): number of turbines
        D (float): rotor diameter [m]
        min_dist (float): enforced minimum spacing between turbine centers
            normalized by rotor diameter
        idx (int, optional): random number generator seed
    
    Args:
        xx (np.array): x-positions of each turbine
        yy (np.array): y-positions of each turbine

    """

    print("Generating wind farm layout.")

    # Verify that boundaries and turbines are supplied
    if not boundaries:
        raise ValueError("Must supply boundaries to generate wind farm.")
    
    if n_turb <= 0:
        raise ValueError("Must supply number of turbines.")

    # Initialize RNG and containers
    if idx != None:
        np.random.seed(idx)

    xx = np.zeros(n_turb)
    yy = np.zeros(n_turb)

    xmin = np.min([tup[0] for tup in boundaries])
    xmax = np.max([tup[0] for tup in boundaries])
    ymin = np.min([tup[1] for tup in boundaries])
    ymax = np.max([tup[1] for tup in boundaries])

    # Generate boundary polygon
    poly = Polygon(boundaries)

    # Generate new positions and check minimum spacing and boundary
    for i in range(n_turb):
        prop_x = np.random.uniform(low=xmin, high=xmax)
        prop_y = np.random.uniform(low=ymin, high=ymax)
        pt = Point(prop_x, prop_y)
        while np.any(np.sqrt((prop_x - xx)**2 + (prop_y - yy)**2) < min_dist*D) or not pt.within(poly):
            prop_x = np.random.uniform(low=xmin, high=xmax)
            prop_y = np.random.uniform(low=ymin, high=ymax)
            pt = Point(prop_x, prop_y)
        xx[i] = prop_x
        yy[i] = prop_y

    return xx, yy

def load_layout(name, boundaries=False):
    """
    TODO: fill in description

    """

    if name == "iea":
        layout_x = np.array([
            2714.43,
            2416.08,
            1496.75,
            1860.65,
            2224.55,
            2588.45,
            1197.40,
            1619.09,
            2040.78,
            2462.46,
            898.05,
            1257.66,
            1617.27,
            1976.87,
            2336.48,
            598.70,
            1001.65,
            1404.60,
            1807.55,
            2210.50,
            299.35,
            750.21,
            1201.07,
            1651.93,
            2102.79,
            0.00,
            415.30,
            830.59,
            1245.88,
            1661.18,
            2076.47,
            ])
        layout_y = np.array([
            4042.95,
            3932.63,
            4020.72,
            3803.55,
            3586.38,
            3369.22,
            3618.82,
            3311.04,
            3003.27,
            2695.49,
            3216.92,
            2918.13,
            2619.34,
            2320.55,
            2021.76,
            2815.01,
            2448.27,
            2081.53,
            1714.78,
            1348.04,
            2413.11,
            1977.72,
            1542.33,
            1106.94,
            671.55,
            2011.21,
            1608.97,
            1206.72,
            804.48,
            402.24,
            0.00,
        ])
        bound = [
            (2714.4, 4049.4),
            (2132.7, 938.8),
            (2092.8, 591.6),
            (2078.9, 317.3),
            (2076.1, 148.5),
            (2076.6, 0.0),
            (2076.5, 6.5),
            (1208.6, 847.0),
            (0.0, 2017.7),
            (1496.7, 4027.2),
            (1531.8, 4006.2),
            (1931.2, 3818.5),
            (2058.3, 3783.6),
            (2192.8, 3792.9),
            (2316.8, 3846.4),
            (2416.0, 3939.1),
            (2528.6, 4089.0),
            (2550.9, 4126.3)
        ]
    
    if boundaries:
        return layout_x, layout_y, bound
    else:
        return layout_x, layout_y

###########################################################################
# Wind rose sampling
###########################################################################

def toolkit_wind_rose(lat, long):
    """
    Sample a wind rose from the WIND Toolkit (copied from FLORIS)

    Args:
        lat (float): latitude of wind farm site (in continental US)
        long (float): longitude of wind farm site (in continental US)
    
    Returns:
        df (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction

    """

    wind_rose = rose.WindRose()
    wd_list = np.arange(0, 360, 1)
    ws_list = np.arange(0, 26, 1)
    df = wind_rose.import_from_wind_toolkit_hsds(
        lat,
        long,
        ht=100,
        wd=wd_list,
        ws=ws_list,
    )
    return df

def load_wind_rose(idx):
    """
    Load a locally-stored wind rose saved to a pickle file.
    See show_wind_roses.py to visualize all wind rose options.

    Args:
        idx (int): index of desired wind rose
    
    Returns:
        df (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction

    """

    print("Generating wind rose.")
    file_name = './wind_roses/wr' + str(idx) + '.p'
    df = pd.read_pickle(file_name)
    
    return df

def resample_wind_direction(df, wd=np.arange(0, 360, 5.0)):
    """
    Resample wind direction bins using new specified bin center values.
    (Copied from FLORIS)

    Args:
        df (pandas.DataFrame): Wind rose DataFrame containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurance of the
                wind conditions in the other columns.

        wd (np.array, optional): List of new wind direction center bins
            (deg). Defaults to np.arange(0, 360, 5.).

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': New wind direction bin center values from wd argument (deg).
            - 'ws': Resampled wind speed bin center values (m/s).
            - 'freq_val': The resampled frequency of occurance of the
                wind conditions in the other columns.
    """

    # Make a copy of incoming dataframe
    df = df.copy(deep=True)

    # Get the wind step
    wd_step = wd[1] - wd[0]

    # Get bin edges
    wd_edges = wd - wd_step / 2.0
    wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

    # Get the overhangs
    negative_overhang = wd_edges[0]
    positive_overhang = wd_edges[-1] - 360.0

    # Potentially wrap high angle direction to negative for correct binning
    tmp = df.wd
    tmp = np.where(tmp < 0.0, tmp + 360.0, tmp)
    tmp = np.where(tmp >= 360.0, tmp - 360.0, tmp)
    df["wd"] = tmp

    if negative_overhang < 0:
        #print("Correcting negative Overhang:%.1f" % negative_overhang)
        df["wd"] = np.where(
            df.wd.values >= 360.0 + negative_overhang,
            df.wd.values - 360.0,
            df.wd.values,
        )

    # Check on other side
    if positive_overhang > 0:
        #print("Correcting positive Overhang:%.1f" % positive_overhang)
        df["wd"] = np.where(
            df.wd.values <= positive_overhang, df.wd.values + 360.0, df.wd.values
        )

    # Cut into bins
    df["wd"] = pd.cut(df.wd, wd_edges, labels=wd)

    # Regroup
    df = df.groupby([c for c in df.columns if c != "freq_val"]).sum()

    # Fill nans
    df = df.fillna(0)

    # Reset the index
    df = df.reset_index()

    # Set to float Re-wrap
    for c in [c for c in df.columns if c != "freq_val"]:
        df[c] = df[c].astype(float)
        df[c] = df[c].astype(float)
    
    tmp = df.wd
    tmp = np.where(tmp < 0.0, tmp + 360.0, tmp)
    tmp = np.where(tmp >= 360.0, tmp - 360.0, tmp)
    df["wd"] = tmp

    return df

def resample_average_ws_by_wd(df):
        """
        Calculate the mean wind speed for each wind direction bin
        and resample the wind rose. (Copied from FLORIS)

    Args:
        df (pandas.DataFrame): Wind rose DataFrame containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurance of the
                wind conditions in the other columns.

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Resampled average wind speed values (m/s).
            - 'freq_val': The resampled frequency of occurance of the
                wind conditions in the other columns.
                
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        ws_avg = []

        for val in df.wd.unique():
            ws_avg.append(
                np.array(
                    df.loc[df["wd"] == val]["ws"] * df.loc[df["wd"] == val]["freq_val"]
                ).sum()
                / df.loc[df["wd"] == val]["freq_val"].sum()
            )

        # Regroup
        df = df.groupby("wd").sum()

        df["ws"] = ws_avg

        # Reset the index
        df = df.reset_index()

        # Set to float
        df["ws"] = df.ws.astype(float)
        df["wd"] = df.wd.astype(float)

        return df


###########################################################################
# Turbine parameter tables
###########################################################################

def ct_lookup(u):
    """
    Look-up table for thrust coefficient of the NREL 5 MW turbine.

    Args:
        u (float): inflow wind speed [m/s]
    
    Returns:
        ct (float): thrust coefficient
    
    """

    ct_table = np.array([1.19187945,1.17284634,1.09860817,1.02889592,0.97373036,
    0.92826162,0.89210543,0.86100905,0.835423,0.81237673,0.79225789,0.77584769,
    0.7629228,0.76156073,0.76261984,0.76169723,0.75232027,0.74026851,0.72987175,
    0.70701647,0.54054532,0.45509459,0.39343381,0.34250785,0.30487242,0.27164979,
    0.24361964,0.21973831,0.19918151,0.18131868,0.16537679,0.15103727,0.13998636,
    0.1289037,0.11970413,0.11087113,0.10339901,0.09617888,0.09009926,0.08395078,
    0.0791188,0.07448356,0.07050731,0.06684119,0.06345518,0.06032267,0.05741999,
    0.05472609])
    u_table = np.arange(2.0,26.0,0.5)
    
    return np.interp(u, u_table, ct_table)

def cp_lookup(u):
    """
    Look-up table for power coefficient of the NREL 5 MW turbine.

    Args:
        u (float): inflow wind speed [m/s]
    
    Returns:
        cp (float): thrust coefficient
    
    """
    
    cp_table = np.array([0.0,0.0,0.1780851,0.28907459,0.34902166,0.3847278,
        0.40605878,0.4202279,0.42882274,0.43387274,0.43622267,0.43684468,0.43657497,
        0.43651053,0.4365612,0.43651728,0.43590309,0.43467276,0.43322955,0.43003137,
        0.37655587,0.33328466,0.29700574,0.26420779,0.23839379,0.21459275,0.19382354,
        0.1756635,0.15970926,0.14561785,0.13287856,0.12130194,0.11219941,0.10311631,
        0.09545392,0.08813781,0.08186763,0.07585005,0.07071926,0.06557558,0.06148104,
        0.05755207,0.05413366,0.05097969,0.04806545,0.04536883,0.04287006,0.04055141])
    u_table = np.arange(2.0,26.0,0.5)

    return np.interp(u, u_table, cp_table)