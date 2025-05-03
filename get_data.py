import dask.dataframe as dd

TICKETER_CSV = "./data/constituents.csv"
STOCKS_PARQUET = "./data/stocks.parquet"
STOCKS_DUCKDB = "./data/stocks.duckdb"

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def get_ticketers():
    """Get ticketers from CSV file."""
    df = dd.read_csv(TICKETER_CSV)
    ticketers = df["Symbol"].compute().tolist()
    return ticketers


def get_data(ticketers, start_date, end_date):
    """Get data from Yahoo Finance."""
    # yfinance doesn't work directly with Dask, so we'll keep pandas here
    return yf.download(ticketers, start=start_date, end=end_date)['Open']

def save_data(data):
    """Save data to Parquet."""
    ddf = dd.from_pandas(data, npartitions=4)
    ddf.to_parquet(STOCKS_PARQUET, engine='pyarrow')

def load_data_from_parquet():
    """Load data from Parquet."""
    return dd.read_parquet(STOCKS_PARQUET)

def load_data_from_duck():
    """Load data from DuckDB."""
    con = duckdb.connect(STOCKS_DUCKDB)
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
    save_data(data)