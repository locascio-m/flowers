
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx

import flowers
from flowers import FlowersModel


def test_regression_aep():
    # Create generic layout
    D = 126.
    layout_x = D * np.array([0.,0.,0.,7.,7.,7.,14.,14.,14.,21.,21.,21.,28.,28.,28.])
    layout_y = D * np.array([0.,7.,14.,0.,7.,14.,0.,7.,14.,0.,7.,14.,0.,7.,14.])

    # Load in wind data
    wind_rose_file = Path(
        Path(flowers.__file__).parent,
        "..",
        "examples",
        "inputs",
        "HKW_wind_rose.csv",
    )
    df = pd.read_csv(wind_rose_file)

    # Setup FLOWERS model
    flowers_model = FlowersModel(df, layout_x, layout_y)

    # Calculate the AEP
    aep = flowers_model.calculate_aep()

    assert aep == approx(350958996034.8524, 1e-3)