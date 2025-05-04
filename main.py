import dask.dataframe as dd
import os

from process_data import process_data
from get_data import get_ticketers, get_data, save_data
from autoencoder import create_autoencoder
import argparse

from GLOBALS import STOCKS_PARQUET, START_DATE, END_DATE, NORMALISED_STOCKS_PARQUET

def main():

    parser = argparse.ArgumentParser(description='Finance data processing')
    parser.add_argument('--load-data', action='store_true', help='Get financial data')
    args = parser.parse_args()

    # Generate model folder if it does not exist
    os.makedirs("models", exist_ok=True)

    if args.load_data:
        # ticketers = get_ticketers()
        # data = get_data(ticketers, START_DATE, END_DATE)
        # save_data(STOCKS_PARQUET, data)
        # process_data()

        create_autoencoder(NORMALISED_STOCKS_PARQUET, sequence_length=10)
    


    



if __name__ == "__main__":
    main()