# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

import floris.tools.wind_rose as rose


###########################################################################
# Wind rose sampling
###########################################################################

def random_layout(boundaries=[], n_turb=0, idx=None, D=126.0):
    """
    Generate a random wind farm layout within the specified boundaries.
    Minimum spacing between turbines is 2D.

    Args:
        boundaries (list(float)): A list of the boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)]
        n_turb (int): Number of turbines
        D (float): Rotor diameter [m]
    """
    print("Generating wind farm layout.")
    if idx != None:
        np.random.seed(idx)
    xx = np.zeros(n_turb)
    yy = np.zeros(n_turb)

    xmin = np.min([tup[0] for tup in boundaries])
    xmax = np.max([tup[0] for tup in boundaries])
    ymin = np.min([tup[1] for tup in boundaries])
    ymax = np.max([tup[1] for tup in boundaries])

    poly = Polygon(boundaries)

    # Generate new positions and check minimum spacing and boundary
    for i in range(n_turb):
        prop_x = np.random.uniform(low=xmin, high=xmax)
        prop_y = np.random.uniform(low=ymin, high=ymax)
        pt = Point(prop_x, prop_y)
        while np.any(np.sqrt((prop_x - xx)**2 + (prop_y - yy)**2) < 2*D) or not pt.within(poly):
            prop_x = np.random.uniform(low=xmin, high=xmax)
            prop_y = np.random.uniform(low=ymin, high=ymax)
            pt = Point(prop_x, prop_y)
        xx[i] = prop_x
        yy[i] = prop_y

    return xx, yy

###########################################################################
# Wind rose sampling
###########################################################################

def toolkit_wind_rose(lat, long):
    """
    Sample a wind rose from the WIND Toolkit (copied from FLORIS)

    Args:
        lat (float): latitude of wind farm site (in range ???)
        long (float): longitude of wind farm site (in range ???)
    
    Returns:
        df (pandas:dataframe): dataframe of wind rose:
            - 'ws': np.array of wind speeds in range [0,26] m/s
            - 'wd': np.array of wind directions in range [0,360) m/s
            - 'freq_val': np.array of frequency for each wind speed and direction
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
    Descriptions of each index:
        ** 2 ** : high speed, two predominant wind directions

    Args:
        idx (int): index of desired wind rose
    
    Returns:
        df (pandas:dataframe): dataframe of wind rose:
            - 'ws': np.array of wind speeds in range [0,26] m/s
            - 'wd': np.array of wind directions in range [0,360) m/s
            - 'freq_val': np.array of frequency for each wind speed and direction
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

    # Need potentially to wrap high angle direction to negative for correct
    # binning
    tmp = df.wd
    tmp = np.where(tmp < 0.0, tmp + 360.0, tmp)
    tmp = np.where(tmp >= 360.0, tmp - 360.0, tmp)
    df["wd"] = tmp

    if negative_overhang < 0:
        print("Correcting negative Overhang:%.1f" % negative_overhang)
        df["wd"] = np.where(
            df.wd.values >= 360.0 + negative_overhang,
            df.wd.values - 360.0,
            df.wd.values,
        )

    # Check on other side
    if positive_overhang > 0:
        print("Correcting positive Overhang:%.1f" % positive_overhang)
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

def ct_lookup(u_infty):
    """Returns thrust coefficient for NREL 5MW turbine for wind speed input"""

    ct_table = np.array([1.19187945,1.17284634,1.09860817,1.02889592,0.97373036,
    0.92826162,0.89210543,0.86100905,0.835423,0.81237673,0.79225789,0.77584769,
    0.7629228,0.76156073,0.76261984,0.76169723,0.75232027,0.74026851,0.72987175,
    0.70701647,0.54054532,0.45509459,0.39343381,0.34250785,0.30487242,0.27164979,
    0.24361964,0.21973831,0.19918151,0.18131868,0.16537679,0.15103727,0.13998636,
    0.1289037,0.11970413,0.11087113,0.10339901,0.09617888,0.09009926,0.08395078,
    0.0791188,0.07448356,0.07050731,0.06684119,0.06345518,0.06032267,0.05741999,
    0.05472609])
    u_table = np.arange(2.0,26.0,0.5)
    return np.interp(u_infty, u_table, ct_table)

def cp_lookup(u_avg):
    """Returns power coefficient for NREL 5MW turbine for wind speed input"""
    
    cp_table = np.array([0.0,0.0,0.1780851,0.28907459,0.34902166,0.3847278,
        0.40605878,0.4202279,0.42882274,0.43387274,0.43622267,0.43684468,0.43657497,
        0.43651053,0.4365612,0.43651728,0.43590309,0.43467276,0.43322955,0.43003137,
        0.37655587,0.33328466,0.29700574,0.26420779,0.23839379,0.21459275,0.19382354,
        0.1756635,0.15970926,0.14561785,0.13287856,0.12130194,0.11219941,0.10311631,
        0.09545392,0.08813781,0.08186763,0.07585005,0.07071926,0.06557558,0.06148104,
        0.05755207,0.05413366,0.05097969,0.04806545,0.04536883,0.04287006,0.04055141])
    u_table = np.arange(2.0,26.0,0.5)
    return np.interp(u_avg, u_table, cp_table)