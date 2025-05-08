import pandas as pd

from GLOBALS import NORMALISED_STOCKS_PARQUET, STOCKS_PARQUET
from get_data import save_data, load_data_from_parquet

def normalise_data(data):
    """
    Normalises the column (z-score)
    """
    return (data - data.mean()) / data.std()

def process_data():
    """
    Process the data by normalising it and dropping NaN values.
    """

    data = load_data_from_parquet(STOCKS_PARQUET)

    data = normalise_data(data)
    
    save_data(NORMALISED_STOCKS_PARQUET, data)

    return data