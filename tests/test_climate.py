import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.climate_prototype import preprocess_data

def test_preprocess_data():
    data = {
        "Year": [2000, 2001, 2002],
        "Temperature_C": [24.98, 25.01, 25.05],
        "Rainfall_mm": [1080, 1120, 950],
        "CO2_Emissions_tonnes": [996890915, 1050000000, 1100000000]
    }

    df = pd.DataFrame(data)
    result = preprocess_data(df)

    assert len(result) == 3
    assert result["Year"].min() == 2000
    assert result["Year"].max() == 2002


def test_no_missing_values():
    data = {
        "Year": [2000, 2001],
        "Temperature_C": [24.98, 25.01],
        "Rainfall_mm": [1080, 1120],
        "CO2_Emissions_tonnes": [996890915, 1050000000]
    }

    df = pd.DataFrame(data)
    result = preprocess_data(df)

    assert result.isnull().sum().sum() == 0