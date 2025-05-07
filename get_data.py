import dask.dataframe as dd
from GLOBALS import TICKETER_CSV, START_DATE, END_DATE, STOCKS_PARQUET, STOCKS_DUCKDB, LOADED_TICKETER_CSV
import yfinance as yf
import duckdb
import os

def get_ticketers():
    """Get ticketers from CSV file."""
    with open(TICKETER_CSV, 'r') as f:
        ticketers = f.read().splitlines()
    return ticketers

def get_loaded_ticketers():
    """Get loaded ticketers from CSV file."""
    with open(LOADED_TICKETER_CSV, 'r') as f:
        ticketers = f.read().splitlines()
    return ticketers

def save_ticketers(ticketers):
    """Save ticketers to CSV file."""
    ticketers_loaded = get_loaded_ticketers()
    ticketers = list(set(ticketers + ticketers_loaded))
    with open(LOADED_TICKETER_CSV, 'w') as f:
        for ticketer in ticketers:
            f.write(f"{ticketer}\n")

def get_data(ticketers, start_date, end_date):
    """Get data from Yahoo Finance."""
    # yfinance doesn't work directly with Dask, so we'll keep pandas here
    loaded_ticketers = get_loaded_ticketers()
    ticketers = list(set(ticketers) - set(loaded_ticketers))

    print(ticketers)

    if ticketers:

        df = yf.download(ticketers, start=start_date, end=end_date)['Open']

        df = df.dropna(axis=1, how='all')

        if not df.empty:
            save_ticketers(list(map(lambda k: k.split(",")[0], df.columns)))
            save_data(STOCKS_PARQUET, df)

    return load_data_from_parquet(STOCKS_PARQUET)

def save_data(filename, new_data):
    """Save data to Parquet."""
    curr_data = load_data_from_parquet(filename)
    # Combine the new data with the existing data
    if curr_data is not None:
        new_data = dd.concat([curr_data, new_data])
    
    os.remove(filename)
    new_data.to_parquet(filename, engine='pyarrow')

def load_data_from_parquet(filename):
    """Load data from Parquet."""
    if not os.path.exists(filename):
        return None

    df = dd.read_parquet(filename)
    df = df.set_index("Date")
    df = df.reindex(sorted(df.columns), axis=1)
    
    return df

# def load_data_from_duck(filename):
#     """Load data from DuckDB."""
#     con = duckdb.connect(filename)
#     df = con.execute("SELECT * FROM stock_prices").fetchdf()
#     return dd.from_pandas(df, npartitions=4)

# def convert_data_to_duck():
#     """Load data from Parquet into DuckDB."""
#     df = dd.read_parquet(STOCKS_PARQUET).compute()
#     df = df.reset_index()
#     con = duckdb.connect(STOCKS_DUCKDB)
#     con.execute("CREATE TABLE stock_prices AS SELECT * FROM df")

if __name__ == "__main__":
    ticketers = get_ticketers()
    data = get_data(ticketers, START_DATE, END_DATE)
    save_data(STOCKS_PARQUET, data)
