import dask.dataframe as dd
from GLOBALS import TICKETER_CSV, START_DATE, END_DATE, STOCKS_PARQUET, STOCKS_DUCKDB
import yfinance as yf
import duckdb

def get_ticketers():
    """Get ticketers from CSV file."""
    df = dd.read_csv(TICKETER_CSV)
    ticketers = df["Symbol"].compute().tolist()
    return ticketers


def get_data(ticketers, start_date, end_date):
    """Get data from Yahoo Finance."""
    # yfinance doesn't work directly with Dask, so we'll keep pandas here
    return yf.download(ticketers, start=start_date, end=end_date)['Open']

def save_data(filename, data):
    """Save data to Parquet."""
    data.to_parquet(filename, engine='pyarrow')

def load_data_from_parquet(filename):
    """Load data from Parquet."""
    df = dd.read_parquet(filename)
    df = df.set_index("Date")
    return df

def load_data_from_duck(filename):
    """Load data from DuckDB."""
    con = duckdb.connect(filename)
    df = con.execute("SELECT * FROM stock_prices").fetchdf()
    return dd.from_pandas(df, npartitions=4)

def convert_data_to_duck():
    """Load data from Parquet into DuckDB."""
    df = dd.read_parquet(STOCKS_PARQUET).compute()
    df = df.reset_index()
    con = duckdb.connect(STOCKS_DUCKDB)
    con.execute("CREATE TABLE stock_prices AS SELECT * FROM df")

if __name__ == "__main__":
    ticketers = get_ticketers()
    data = get_data(ticketers, START_DATE, END_DATE)
    save_data(STOCKS_PARQUET, data)
